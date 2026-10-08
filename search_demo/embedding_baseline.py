"""Standard BGE retrieval baseline with the model's recommended query instruction."""
import argparse,json
from pathlib import Path
import numpy as np
from experiment import api
from jevector import atomic_json,digest
from search_demo.run import metrics
MODEL='@cf/baai/bge-small-en-v1.5'
PREFIX='Represent this sentence for searching relevant passages: '
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--account',required=True);args=p.parse_args();data=json.loads(Path('search_demo/data.json').read_text());texts=[d['title']+'\n'+d['body'] for d in data['documents']]+[PREFIX+q['text'] for q in data['queries']]
 fingerprint=digest({'model':MODEL,'texts':texts,'pooling':'cls'});path=Path('search_demo/runs/bge-small-raw.json')
 if path.exists():
  record=json.loads(path.read_text())
  if record['fingerprint']!=fingerprint:raise ValueError('Stale embedding cache')
 else:
  record={'model':MODEL,'fingerprint':fingerprint,'response':api(f'accounts/{args.account}/ai/run/{MODEL}',{'text':texts,'pooling':'cls'})};atomic_json(path,record)
 a=np.asarray(record['response']['data'],dtype=np.float32)
 if a.shape!=(len(texts),384) or not np.isfinite(a).all():raise ValueError('Invalid BGE output')
 a=a/np.linalg.norm(a,axis=1,keepdims=True);n=len(data['documents']);rows=[]
 for q,v in zip(data['queries'],a[n:]):
  scores=a[:n]@v;order=np.argsort(-scores);rows.append({**q,'bge':[data['documents'][i]['id'] for i in order[:5]]})
 result={'model':MODEL,'dimensions':384,'pooling':'cls','query_prefix':PREFIX,'metrics':metrics(rows,'bge'),'rows':rows};atomic_json('search_demo/runs/bge-small-results.json',result);print(result['metrics'])
