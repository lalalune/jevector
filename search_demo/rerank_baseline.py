"""Rerank the first 10 BM25 results with saved BGE-small cosine scores."""
import json
from pathlib import Path
import numpy as np
from jevector import atomic_json,digest
from search_demo.embedding_baseline import MODEL,PREFIX
from search_demo.run import bm25,metrics

def rerank(candidates,scores,k=5):
 return sorted(candidates,key=lambda doc:-scores[doc])[:k]

def main():
 data=json.loads(Path('search_demo/data.json').read_text())
 record=json.loads(Path('search_demo/runs/bge-small-raw.json').read_text())
 texts=[d['title']+'\n'+d['body'] for d in data['documents']]+[PREFIX+q['text'] for q in data['queries']]
 if record['fingerprint']!=digest({'model':MODEL,'texts':texts,'pooling':'cls'}):raise ValueError('Stale BGE cache')
 a=np.asarray(record['response']['data'],dtype=np.float32)
 if a.shape!=(len(texts),384) or not np.isfinite(a).all() or np.any(np.linalg.norm(a,axis=1)==0):raise ValueError('Invalid embeddings')
 a=a/np.linalg.norm(a,axis=1,keepdims=True);n=len(data['documents']);rows=[]
 for q,v in zip(data['queries'],a[n:]):
  scores=dict(zip([d['id'] for d in data['documents']],map(float,a[:n]@v)))
  candidates=bm25(data['documents'],q['text'])[:10]
  rows.append({**q,'candidates':candidates,'reranked':rerank(candidates,scores)})
 result={'model':MODEL,'dimensions':384,'candidate_count':10,'method':'BM25 top 10, then BGE-small cosine reranking','fixture_hash':digest(data),'metrics':metrics(rows,'reranked'),'rows':rows}
 atomic_json('search_demo/runs/bge-bm25-results.json',result)
 print(result['metrics'])
if __name__=='__main__':main()
