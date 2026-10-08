"""Evaluate BM25 and BGE-small on the same paired profile views as Clef and Jev."""
import argparse,json
from pathlib import Path
import numpy as np
from experiment import api
from jevector import atomic_json,digest
from search_demo.embedding_baseline import MODEL,PREFIX
from search_demo.run import bm25

def summarize(rankings,people):
 rows=[{'query':p['id'],'ranking':r,'rank':r.index(p['id'])+1} for p,r in zip(people,rankings)]
 return {'top1':sum(r['rank']==1 for r in rows)/len(rows),'rows':rows}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--account',required=True);args=parser.parse_args();out={}
 for name,path in [('original','data/synthetic.json'),('additional','data/fresh-stress.json')]:
  data=json.loads(Path(path).read_text());people=data['people'];n=len(people)
  # Serialize only supplied profile state; exclude person IDs and expected labels.
  views={v:[json.dumps(p['views'][v],ensure_ascii=False) for p in people] for v in ('a','b')}
  docs=[{'id':p['id'],'title':'','body':s} for p,s in zip(people,views['a'])]
  lexical=[bm25(docs,q) for q in views['b']]
  texts=views['a']+[PREFIX+q for q in views['b']];vectors=[]
  for offset in range(0,len(texts),16):
   chunk=texts[offset:offset+16];fingerprint=digest({'model':MODEL,'texts':chunk,'pooling':'cls'})
   cache=Path('runs/profile-baselines/cache')/(fingerprint+'.json')
   if cache.exists():record=json.loads(cache.read_text())
   else:
    record={'fingerprint':fingerprint,'model':MODEL,'response':api(f'accounts/{args.account}/ai/run/{MODEL}',{'text':chunk,'pooling':'cls'})};atomic_json(cache,record)
   vectors.extend(record['response']['data'])
  a=np.asarray(vectors,dtype=np.float32)
  if a.shape!=(2*n,384) or not np.isfinite(a).all() or np.any(np.linalg.norm(a,axis=1)==0):raise ValueError('Invalid BGE response')
  a/=np.linalg.norm(a,axis=1,keepdims=True)
  rankings=[]
  for v in a[n:]:rankings.append([people[i]['id'] for i in np.argsort(-(a[:n]@v),kind='stable')])
  out[name]={'dataset_hash':digest(data),'people':n,'bm25':summarize(lexical,people),'bge':summarize(rankings,people)}
  print(name,{k:out[name][k]['top1'] for k in ('bm25','bge')},flush=True)
 atomic_json('runs/profile-baselines/results.json',{'model':MODEL,'dimensions':384,'pooling':'cls','query_prefix':PREFIX,'serialization':'JSON profile state, without IDs or expected labels','results':out})
if __name__=='__main__':main()
