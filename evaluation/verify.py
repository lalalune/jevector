"""Verify saved evaluation artifacts against source data and stored rankings."""
import json
from pathlib import Path
import numpy as np
from jevector import digest
from evaluation.common import metrics,rank
from evaluation.science_schema import SCHEMA_HASH

def main():
 data=json.loads(Path('runs/evaluation/public-data.json').read_text());r=json.loads(Path('runs/evaluation/public-results.json').read_text())
 assert r['data_hash']==digest(data) and r['schema_hash']==SCHEMA_HASH
 ids={d['id'] for d in data['documents']};assert len(ids)==500
 queries=data['queries'];assert len(queries)==80
 assert sum(q['split']=='calibration' for q in queries)==20
 assert sum(q['split']=='test' and bool(q['relevant']) for q in queries)==45
 assert sum(q['split']=='test' and not q['relevant'] for q in queries)==15
 for q in queries:
  assert set(q['relevant'])<=ids
  assert not set(q['withheld_relevant'])&ids
 for name,method in r['methods'].items():
  rows=method['rows'];assert [x['id'] for x in rows]==[q['id'] for q in queries]
  for row,q in zip(rows,queries):
   assert row['relevant']==q['relevant'] and row['split']==q['split']
   assert len(row['ranking'])==len(set(row['ranking'])) and set(row['ranking'])<=ids
   assert row['accepted']==(row['score']>=method['metrics']['threshold'])
  answered=[x for x in rows if x['split']=='test' and x['relevant']]
  measured=metrics([x['ranking'] for x in answered],[x['relevant'] for x in answered])
  for key,value in measured.items():assert abs(value-method['metrics'][key])<1e-12,(name,key)
 for provider in ('clef','jev'):
  v=json.loads(Path(f'runs/evaluation/{provider}-public-vectors.json').read_text());assert v['data_hash']==digest(data)
  a=np.array(v['documents'],dtype=np.float32);b=np.array(v['queries'],dtype=np.float32)
  assert a.shape==(500,64) and b.shape==(80,64) and np.isfinite(a).all() and np.isfinite(b).all()
  a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-9);b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-9)
  order=[rank(s,[d['id'] for d in data['documents']])[:10] for s in b@a.T]
  assert order==[x['ranking'] for x in r['methods'][provider.capitalize()+' 64']['rows']]
 audit=json.loads(Path('runs/evaluation/profile-audit.json').read_text())
 for group,file in [('original','data/synthetic.json'),('additional','data/fresh-stress.json')]:
  people=json.loads(Path(file).read_text())['people'];ids=[p['id'] for p in people]
  assert audit[group]['people']==len(ids)
  for name,variant in audit[group]['variants'].items():
   assert all(len(order)==len(ids) and set(order)==set(ids) for order in variant['rankings'])
   measured=metrics(variant['rankings'],[[i] for i in ids])
   assert measured==variant['metrics'],(group,name)
 print('Verified public splits, excluded positives, all rankings and scores, both provider matrices, and all profile baseline variants.')
if __name__=='__main__':main()
