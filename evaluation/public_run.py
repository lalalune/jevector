"""Evaluate a frozen 64-question rubric and reference retrieval methods on public data."""
import argparse,concurrent.futures,getpass,json,time,math
from pathlib import Path
import numpy as np
from evaluation.common import embed,cached_call,rank,rrf,metrics,PREFIX,tokenizer,reranker_tokenizer
from evaluation.science_schema import questions,SCHEMA_HASH
from search_demo.run import bm25
from jevector import Client,atomic_json,digest

def evaluate(orders,scores,queries):
 cal=[i for i,q in enumerate(queries) if q['split']=='calibration'];test=[i for i,q in enumerate(queries) if q['split']=='test'];answered=[i for i in test if queries[i]['relevant']]
 # Threshold maximizes balanced answerability accuracy on calibration only.
 candidates=sorted(set(float(scores[i]) for i in cal))+[max(scores[i] for i in cal)+1e-6]
 def balanced(threshold,indices):
  yes=[i for i in indices if queries[i]['relevant']];no=[i for i in indices if not queries[i]['relevant']]
  return .5*(sum(scores[i]>=threshold for i in yes)/len(yes)+sum(scores[i]<threshold for i in no)/len(no))
 threshold=max(candidates,key=lambda x:(balanced(x,cal),x))
 m=metrics([orders[i] for i in answered],[queries[i]['relevant'] for i in answered]);n=len(answered);p=m['top1'];z=1.96
 center=(p+z*z/(2*n))/(1+z*z/n);radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
 no=[i for i in test if not queries[i]['relevant']]
 m.update({'answered_queries':n,'top1_wilson95':[center-radius,center+radius],'threshold':threshold,'answerability_balanced_accuracy':balanced(threshold,test),'unanswerable_false_accept_rate':sum(scores[i]>=threshold for i in no)/len(no),'answered_accept_rate':sum(scores[i]>=threshold for i in answered)/len(answered)})
 return {'metrics':m,'rows':[{'id':q['id'],'split':q['split'],'relevant':q['relevant'],'ranking':o[:10],'score':float(score),'accepted':bool(score>=threshold)} for q,o,score in zip(queries,orders,scores)]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--account',required=True);p.add_argument('--providers',nargs='+',default=['clef']);p.add_argument('--prompt-key',action='store_true');args=p.parse_args();key=getpass.getpass('Jev key: ') if args.prompt_key else None
 data=json.loads(Path('runs/evaluation/public-data.json').read_text());assert data['schema_hash']==SCHEMA_HASH
 docs=data['documents'];queries=data['queries'];ids=[d['id'] for d in docs];texts=[d['title']+'\n'+d['body'] for d in docs];qtexts=[q['text'] for q in queries]
 out={'data_hash':digest(data),'schema_hash':SCHEMA_HASH,'documents':len(docs),'queries':len(queries),'methods':{}};dest='runs/evaluation/public-results.json'
 if Path(dest).exists():
  previous=json.loads(Path(dest).read_text())
  if previous['data_hash']==out['data_hash'] and previous['schema_hash']==SCHEMA_HASH:out=previous
 def save(name,value):out['methods'][name]=value;atomic_json(dest,out);print(name,value['metrics'],flush=True)
 start=time.perf_counter()
 # BM25 top-score computation shares the exact reference implementation.
 from evaluation.lexical import BM25
 lexical=BM25(texts);lexscores=np.asarray([lexical.score(q) for q in qtexts]);lex=[rank(s,ids) for s in lexscores]
 save('BM25',{**evaluate(lex,lexscores.max(axis=1),queries),'local_total_seconds':time.perf_counter()-start})
 a,da=embed(args.account,texts);b,db=embed(args.account,qtexts,prefix=PREFIX);scores=b@a.T;dense=[rank(s,ids) for s in scores]
 save('BGE-small 384',{**evaluate(dense,scores.max(axis=1),queries),'document_embedding':da,'query_embedding':db})
 fused=[rrf([x,y]) for x,y in zip(lex,dense)]
 fusion_scores=[]
 for i,o in enumerate(fused):fusion_scores.append(sum(1/(61+r.index(o[0])) for r in (lex[i],dense[i])))
 save('BM25 + BGE RRF',evaluate(fused,fusion_scores,queries))
 t=reranker_tokenizer();rerank_truncations=0
 def rerank_one(i):
  candidates=fused[i][:20];query=qtexts[i];budget=max(16,500-len(t.encode(query).ids));contexts=[];truncated=0
  for docid in candidates:
   text=texts[ids.index(docid)];enc=t.encode(text,add_special_tokens=False)
   if len(enc.ids)>budget:text=text[:enc.offsets[budget-1][1]];truncated+=1
   if len(t.encode(query,text).ids)>512:raise ValueError('Rerank pair exceeds 512 tokens')
   contexts.append({'text':text})
  r,hit=cached_call(args.account,'@cf/baai/bge-reranker-base',{'query':query,'contexts':contexts,'top_k':len(contexts)})
  response=r['response'];rows=response['response'] if isinstance(response,dict) else response
  rows=sorted(rows,key=lambda row:-row['score'])
  return [candidates[row['id']] for row in rows],float(rows[0]['score']),r['elapsed_seconds'],truncated,hit
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:reranked=list(pool.map(rerank_one,range(len(queries))))
 save('RRF + cross-encoder',{**evaluate([r[0] for r in reranked],[r[1] for r in reranked],queries),'candidate_count':20,'truncated_candidate_inputs':sum(r[3] for r in reranked),'cache_hits':sum(r[4] for r in reranked),'recorded_query_seconds_p50':float(np.median([r[2] for r in reranked])),'recorded_query_seconds_p95':float(np.percentile([r[2] for r in reranked],95))})
 for provider in args.providers:
  client=Client(provider,args.account,key=key,cache='runs/evaluation/question-cache');qs=questions()
  def extract(text):
   r=client.call({'input':text},qs);answers=r['response']['answers'];v=[answers[k]['probabilities']['yes']/sum(answers[k]['probabilities'].values()) for k in qs]
   return v,r['elapsed_seconds'],r['response'].get('usage',{}).get('input_tokens',0)
  start=time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
   vectors=[]
   for i,v in enumerate(pool.map(extract,texts+qtexts)):
    vectors.append(v)
    if (i+1)%50==0:print(provider,'encoded',i+1,'/',len(texts+qtexts),flush=True)
  a=np.asarray([v[0] for v in vectors[:len(docs)]],dtype=np.float32);b=np.asarray([v[0] for v in vectors[len(docs):]],dtype=np.float32)
  an=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-9);bn=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-9);scores=bn@an.T
  orders=[rank(s,ids) for s in scores];stats={'index_input_tokens':sum(v[2] for v in vectors[:len(docs)]),'query_input_tokens':sum(v[2] for v in vectors[len(docs):]),'recorded_query_seconds_p50':float(np.median([v[1] for v in vectors[len(docs):]])),'recorded_query_seconds_p95':float(np.percentile([v[1] for v in vectors[len(docs):]],95)),'wall_seconds_this_run':time.perf_counter()-start}
  atomic_json(f'runs/evaluation/{provider}-public-vectors.json',{'schema_hash':SCHEMA_HASH,'data_hash':digest(data),'documents':a.tolist(),'queries':b.tolist(),'stats':stats})
  save(provider.capitalize()+' 64',{**evaluate(orders,scores.max(axis=1),queries),**stats})
  # Cheap retrieval ablation on identical extracted features; no additional model.
  tags=(a>=.5).astype(float);qtags=(b>=.5).astype(float);overlap=(qtags@tags.T)/np.maximum(qtags.sum(axis=1,keepdims=True),1)
  save(provider.capitalize()+' binary tags',evaluate([rank(s,ids) for s in overlap],overlap.max(axis=1),queries))
 print('Public evaluation saved:',dest)
if __name__=='__main__':main()
