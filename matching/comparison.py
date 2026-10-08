"""Comparable partner-retrieval metrics from saved decision and BGE outputs."""
import json, math
from pathlib import Path
import numpy as np
from jevector import atomic_json, digest
from matching.ann import ProfileIndex, RETRIEVAL_POLICY
from matching.engine import POLICY_HASH
from evaluation.common import embed, plain
from evaluation.lexical import BM25

def metrics(data, orders, split):
 people={p['id']:p for p in data['people']}
 queries=[q for q in data['query_ids'] if people[q]['kind']=='paired' and (split=='all' or people[q]['split']==split)]
 totals=dict(top1=0.,top5=0.,top10=0.,mrr10=0.,ndcg10=0.,tie_top1=0.,tie_top5=0.)
 for q in queries:
  rows=orders[q];target=people[q]['partner'];pos=next((i for i,r in enumerate(rows) if r['id']==target),None)
  if pos is None:continue
  for k in (1,5,10):totals[f'top{k}']+=pos<k
  if pos<10:totals['mrr10']+=1/(pos+1);totals['ndcg10']+=1/math.log2(pos+2)
  score=rows[pos]['score'];above=sum(r['score']>score+1e-7 for r in rows);tied=sum(abs(r['score']-score)<=1e-7 for r in rows)
  for k in (1,5):totals[f'tie_top{k}']+=min(max(k-above,0),tied)/tied
 return {'queries':len(queries),**{k:v/len(queries) for k,v in totals.items()}}

def main():
 data=json.loads(Path('matching/qualification-data.json').read_text());people=data['people'];ids=[p['id'] for p in people];positions={k:i for i,k in enumerate(ids)}
 result={'data_hash':digest(data),'policy_hash':POLICY_HASH,'retrieval_policy':RETRIEVAL_POLICY,'retrieval':{},'pipeline':{},'rankings':{}}
 def save(section,name,orders):
  result[section][name]={split:metrics(data,orders,split) for split in ('development','validation','all')};result['rankings'][section+'/'+name]=orders
 for provider in ('clef','jev'):
  for size in (64,256):
   vectors={k:json.loads(Path(f'runs/matching-v2/{provider}-{size}/{k}.json').read_text()) for k in ids}
   if any(vectors[p['id']]['input_hash']!=digest(p['state']) for p in people):raise ValueError('Stale model input; rerun qualification')
   index=ProfileIndex(vectors);raw={};pipeline={}
   for q in data['query_ids']:
    candidates=index.candidates(q,32);query=index.query_vectors[index.positions[q]]
    raw[q]=sorted([{'id':k,'score':float(query@index.documents[index.positions[k]])} for k in candidates],key=lambda r:(-r['score'],r['id']))
    pipeline[q]=index.search(q,32,32)['results']
   name=f'{provider.capitalize()} {size}';save('retrieval',name,raw);save('pipeline',name,pipeline)
 own=[p['state']['about_me']+'\nDeclared facts: '+plain(p['state'].get('facts',{})) for p in people]
 wants=[plain(p['state']['looking_for'])+'\nMandatory rules: '+json.dumps(p['state'].get('rules',[])) for p in people]
 # Reuse exact cached requests from qualification. No network access in this report command.
 import evaluation.common as common
 def no_network(*args,**kwargs):raise RuntimeError('Missing saved BGE response. Run matching.qualification --baselines-only first.')
 common.api=no_network
 a,am=embed('',own);b,bm=embed('',wants);result['embedding_cache']={'own':am,'wants':bm}
 lexical=BM25(own);lex=np.asarray([lexical.score(q) for q in wants]);lex/=np.maximum(lex.max(axis=1,keepdims=True),1e-9);lex=(lex+lex.T)/2
 dense=b@a.T;dense=(dense+dense.T)/2
 def ordered(matrix,pools=None):
  return {q:sorted([{'id':k,'score':float(matrix[positions[q],positions[k]])} for k in (pools[q] if pools else ids) if k!=q],key=lambda r:(-r['score'],r['id'])) for q in data['query_ids']}
 orders=ordered(lex);methods={'BM25':orders,'BGE-small 384':ordered(dense),'BM25 + BGE-small':ordered(dense,{q:[r['id'] for r in rows[:10]] for q,rows in orders.items()})}
 for name,rows in methods.items():save('retrieval',name,rows);save('pipeline',name,rows)
 atomic_json('runs/matching-v2/comparison.json',result)
 for name,m in result['retrieval'].items():print(name,m['validation'])
if __name__=='__main__':main()
