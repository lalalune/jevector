"""Build and evaluate live query/document encoders on the support search fixture."""
import argparse,collections,concurrent.futures,getpass,json,math,re
from pathlib import Path
import numpy as np
from jevector import Client,atomic_json,digest
from search_demo.engine import encode,SearchIndex
from search_demo.schema import dimensions,questions,VERSION

def bm25(documents,query):
 def tokens(s):return re.findall(r'[a-z0-9]+',s.lower())
 corpus=[tokens(d['title']+' '+d['body']) for d in documents];avg=sum(map(len,corpus))/len(corpus);df=collections.Counter(t for doc in corpus for t in set(doc));scores=[]
 for doc in corpus:
  tf=collections.Counter(doc);score=0
  for term in tokens(query):
   n=df[term];idf=math.log(1+(len(corpus)-n+.5)/(n+.5));f=tf[term]
   score+=idf*f*2.2/(f+1.2*(.25+.75*len(doc)/avg))
  scores.append(score)
 return [documents[i]['id'] for i in sorted(range(len(documents)),key=lambda i:-scores[i])]

def metrics(rows,field):
 ranks=[]
 for row in rows:
  ranking=row[field];relevant=set(row['relevant']);ranks.append(next((i+1 for i,k in enumerate(ranking) if k in relevant),None))
 return {'top1':sum(r==1 for r in ranks)/len(rows),'recall_at_3':sum(r is not None and r<=3 for r in ranks)/len(rows),'mrr_at_5':sum(1/r if r and r<=5 else 0 for r in ranks)/len(rows)}

def run(provider,size,args,key):
 data=json.loads(Path('search_demo/data.json').read_text());folder=Path('search_demo/runs')/f'{provider}-{size}';client=Client(provider,args.account,key=key,cache='search_demo/cache');jobs=[]
 for d in data['documents']:jobs.append(('doc_'+d['id'],d['title']+'\n'+d['body'],'document'))
 for q in data['queries']:
  jobs += [('query_'+q['id'],q['text'],'query'),('symmetric_'+q['id'],q['text'],'document')]
 vectors={}
 def work(job):
  name,text,role=job;v=encode(client,text,size,role);atomic_json(folder/'vectors'/(name+'.json'),v);return name,v
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for f in concurrent.futures.as_completed([pool.submit(work,j) for j in jobs]):
   name,v=f.result();vectors[name]=v;print(provider,size,len(vectors),'/',len(jobs),name,flush=True)
 index=SearchIndex(folder/'index');index.build(data['documents'],[vectors['doc_'+d['id']] for d in data['documents']]);index=SearchIndex(folder/'index').load();rows=[]
 for q in data['queries']:
  query=vectors['query_'+q['id']];ranked=index.search(query,k=5);symmetric=index.search(vectors['symmetric_'+q['id']],k=5,allow_document_role=True);exact=index.exact(query,5)
  rows.append({**q,'asymmetric':[r['document_id'] for r in ranked],'symmetric':[r['document_id'] for r in symmetric],'exact':exact,'bm25':bm25(data['documents'],q['text'])[:5],'results':ranked,'query_dimensions':sorted([{'goal':d['goal'],'facet':d['facet'],'value':v} for d,v in zip(dimensions(size),query['values'])],key=lambda r:-r['value'])[:8]})
 tie_valid=[]
 for row in rows:
  q=np.asarray(vectors['query_'+row['id']]['values'],dtype=np.float32)
  scores=(index.a@q)/(np.linalg.norm(index.a,axis=1)*np.linalg.norm(q));cutoff=float(np.sort(scores)[-5]);lookup={d['id']:i for i,d in enumerate(data['documents'])}
  tie_valid.append(all(float(scores[lookup[k]])>=cutoff-1e-6 for k in row['asymmetric']))
 summary={'hnsw_top5_valid_including_ties':sum(tie_valid)/len(tie_valid),'provider':provider,'dimensions':size,'schema_version':VERSION,'fixture_hash':digest(data),'models':sorted({m for v in vectors.values() for m in v['models']}),'metrics':{field:metrics(rows,field) for field in ('asymmetric','symmetric','bm25')},'hnsw_exact_top5_set_recall':sum(len(set(r['asymmetric'])&set(r['exact']))/5 for r in rows)/len(rows),'hnsw_exact_top1_agreement':sum(r['asymmetric'][0]==r['exact'][0] for r in rows)/len(rows),'input_tokens':sum(v['input_tokens'] for v in vectors.values()),'rows':rows}
 atomic_json(folder/'results.json',summary);atomic_json(folder/'schema.json',{'version':VERSION,'dimensions':dimensions(size),'query_questions':questions(size,'query'),'document_questions':questions(size,'document')});print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--provider',choices=['clef','jev','both'],required=True);p.add_argument('--dimensions',choices=['64','256','both'],default='both');p.add_argument('--account');p.add_argument('--prompt-key',action='store_true');args=p.parse_args();key=getpass.getpass('Jev API key (hidden): ') if args.prompt_key else None
 for provider in (['clef','jev'] if args.provider=='both' else [args.provider]):
  for size in ([64,256] if args.dimensions=='both' else [int(args.dimensions)]):run(provider,size,args,key)
