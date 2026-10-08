"""Reproduce baseline scores and test input-format sensitivity without replacing the benchmark."""
import argparse,collections,math,re,json
from pathlib import Path
import numpy as np
from evaluation.common import embed,plain,rank,rrf,PREFIX,BGE,chunks,tokenizer,CACHE
from evaluation.lexical import BM25
from jevector import atomic_json,digest
from matching.run import evaluate
from matching.schema import SCHEMA_HASH

def natural_preferences(state):
 phrases={'prefer':'I prefer a person who ','require':'A partner must be someone who ','exclude':'A partner must not be someone who '}
 return ' '.join(phrases[channel]+attribute+'.' for channel in phrases for attribute in state['looking_for'][channel])

def natural_self(text):
 # Rewrite only known sentence wrappers. Do not read oracle fields or resolve attributes.
 rules=[(r'This fits me: ([^.]+)\.',r'This person \1.'),(r'A description of me is: ([^.]+)\.',r'This person \1.'),(r'About myself: ([^.]+) is true\.',r'This person \1.'),(r'This does not fit me: ([^.]+)\.',r'It is false that this person \1.'),(r'I reject this description of myself: ([^.]+)\.',r'It is false that this person \1.'),(r'About myself: ([^.]+) is false\.',r'It is false that this person \1.')]
 for pattern,replacement in rules:text=re.sub(pattern,replacement,text)
 return text

def scalar_bm25(corpus,queries):
 docs=[re.findall('[a-z0-9]+',x.lower()) for x in corpus];n=len(docs);avg=sum(map(len,docs))/n;freq=[collections.Counter(d) for d in docs];df=collections.Counter(t for d in docs for t in set(d));out=[]
 for query in queries:
  qt=collections.Counter(re.findall('[a-z0-9]+',query.lower()));row=[]
  for tokens,tf in zip(docs,freq):
   row.append(sum(count*math.log1p((n-df[term]+.5)/(df[term]+.5))*tf[term]*2.2/(tf[term]+1.2*(.25+.75*len(tokens)/avg)) for term,count in qt.items()))
  out.append(row)
 return np.asarray(out)

def orders(scores,ids,queries):
 lookup={key:i for i,key in enumerate(ids)}
 return {q:[p for p in rank(scores[lookup[q]],ids) if p!=q] for q in queries}

def main():
 p=argparse.ArgumentParser();p.add_argument('--account',required=True);p.add_argument('--format-tests',action='store_true');args=p.parse_args()
 data=json.loads(Path('matching/data.json').read_text());saved=json.loads(Path('runs/matching/results.json').read_text());assert saved['schema_hash']==SCHEMA_HASH and saved['data_hash']==digest(data)
 people=data['people'];ids=[p['id'] for p in people];lookup={k:i for i,k in enumerate(ids)};own=[p['state']['about_me'] for p in people];wants=[plain(p['state']['looking_for']) for p in people]
 report={'data_hash':digest(data),'schema_hash':SCHEMA_HASH,'checks':{},'methods':{}}
 scorer=BM25(own);raw=np.array([scorer.score(q) for q in wants]);independent=scalar_bm25(own,wants);np.testing.assert_allclose(raw,independent,rtol=1e-12,atol=1e-12)
 report['checks']['bm25_max_absolute_error']=float(np.abs(raw-independent).max())
 def bm_scores(corpus,queries,exclude_self_max=False):
  model=BM25(corpus);matrix=np.array([model.score(q) for q in queries]);norm=matrix.copy()
  if exclude_self_max:np.fill_diagonal(norm,0)
  matrix/=np.maximum(norm.max(axis=1,keepdims=True),1e-9);return (matrix+matrix.T)/2
 def record(name,matrix,expected=None):
  ranks=orders(matrix,ids,data['query_ids']);result=evaluate(data,ranks)
  if expected:
   assert result['metrics']==saved['methods'][expected]['metrics']
   assert result['rows']==saved['methods'][expected]['rows']
  report['methods'][name]=result;print(name,result['metrics']['partner_top1'],flush=True);return ranks
 bm=bm_scores(own,wants);bm_ranks=record('BM25 original',bm,'BM25 reciprocal')
 tie_rows=[]
 for person in people:
  if person['kind']!='paired':continue
  row=bm[lookup[person['id']]].copy();row[lookup[person['id']]]=-np.inf;score=row[lookup[person['partner']]];ties=int(np.sum(np.isclose(row,score,rtol=0,atol=1e-12)));higher=int(np.sum(row>score+1e-12))
  tie_rows.append({'id':person['id'],'strictly_higher':higher,'partner_score_ties':ties,'expected_top1_credit':1/ties if higher==0 else 0})
 report['checks']['bm25_ties']={'rows':tie_rows,'expected_correct_top1_under_random_ties':sum(r['expected_top1_credit'] for r in tie_rows)}
 record('BM25 exclude self from normalization',bm_scores(own,wants,True))
 docs,docmeta=embed(args.account,own);report['checks']['document_embedding']=docmeta;report['checks']['norm_max_error']=float(np.abs(np.linalg.norm(docs,axis=1)-1).max())
 for prefix,label in [('', 'no prefix'),(PREFIX,'query prefix')]:
  query,qmeta=embed(args.account,wants,prefix=prefix);matrix=query@docs.T;matrix=(matrix+matrix.T)/2
  # Check the matrix orientation using direct two-way cosine for every evaluated pair.
  error=0.
  for q in data['query_ids']:
   i=lookup[q]
   for j in range(len(ids)):
    expected=(float(np.dot(query[i].astype(float),docs[j].astype(float)))/float(np.linalg.norm(query[i])*np.linalg.norm(docs[j]))+float(np.dot(query[j].astype(float),docs[i].astype(float)))/float(np.linalg.norm(query[j])*np.linalg.norm(docs[i])))/2
    error=max(error,abs(float(matrix[i,j])-expected))
  assert error<1e-5
  report['checks']['cosine_max_error_'+label]=error;report['checks']['query_embedding_'+label]=qmeta
  ranked=record('BGE original '+label,matrix,'BGE reciprocal, '+label)
  # Validate actual API cache metadata and pooling, not just the requested setting.
  t=tokenizer();inputs=[prefix+part for text in wants for part in chunks(text,t)]
  for start in range(0,len(inputs),16):
   key=digest({'model':BGE,'body':{'text':inputs[start:start+16],'pooling':'cls'}});cache=json.loads((CACHE/(key+'.json')).read_text());assert cache['request_hash']==key and cache['model']==BGE and cache['response']['pooling']=='cls'
  if not prefix:
   reranked={q:sorted(bm_ranks[q][:10],key=lambda pid:(-float(matrix[lookup[q],lookup[pid]]),pid)) for q in data['query_ids']}
   assert evaluate(data,reranked)=={k:saved['methods']['BM25 top-10 + BGE rerank'][k] for k in ('metrics','rows')}
   fused={q:rrf([bm_ranks[q],ranked[q]]) for q in data['query_ids']}
   assert evaluate(data,fused)=={k:saved['methods']['BM25 + BGE RRF'][k] for k in ('metrics','rows')}
 report['checks']['rerank_and_rrf_reproduced']=True
 if args.format_tests:
  textqueries=[natural_preferences(p['state']) for p in people];textdocs=[natural_self(x) for x in own]
  for label,corpus,document_vectors in [('explicit query',own,docs),('explicit query and rewritten self',textdocs,None)]:
   record('BM25 '+label,bm_scores(corpus,textqueries))
   if document_vectors is None:document_vectors,_=embed(args.account,corpus)
   for prefix,name in [('', 'no prefix'),(PREFIX,'query prefix')]:
    query,_=embed(args.account,textqueries,prefix=prefix);m=query@document_vectors.T;record('BGE '+label+', '+name,(m+m.T)/2)
 # Fresh, simple semantic controls also check that API batch order is preserved.
 control_docs=['A chess club meets to play board games and study chess openings.','A cooking class teaches bread baking and preparing meals.','A hiking group walks mountain trails and camps outdoors.']
 control_queries=['Where can I play chess?','I want to learn how to bake bread.','I want to walk mountain trails.']
 dv,_=embed(args.account,control_docs);qv,_=embed(args.account,control_queries);reverse,_=embed(args.account,list(reversed(control_docs)))
 chosen=np.argmax(qv@dv.T,axis=1).tolist();assert chosen==[0,1,2]
 agreement=np.sum(dv*reverse[::-1],axis=1);assert np.all(agreement>.999)
 report['checks']['semantic_control_top1']=chosen;report['checks']['batch_order_cosines']=agreement.tolist()
 atomic_json('runs/matching/baseline-audit.json',report)
 print('Saved runs/matching/baseline-audit.json')
if __name__=='__main__':main()
