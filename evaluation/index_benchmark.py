"""Mechanical HNSW recall/latency test. Random vectors are not relevance evidence."""
import json,time,platform
from pathlib import Path
import hnswlib,numpy as np
from jevector import atomic_json

def main():
 rng=np.random.default_rng(42);rows=[]
 for dims in (64,256,384):
  for count in (500,5000,20000):
   a=rng.random((count,dims),dtype=np.float32);a/=np.linalg.norm(a,axis=1,keepdims=True)
   q=rng.random((100,dims),dtype=np.float32);q/=np.linalg.norm(q,axis=1,keepdims=True)
   exact=[];times=[]
   for v in q:
    start=time.perf_counter();scores=a@v;exact.append(np.argsort(-scores)[:10]);times.append((time.perf_counter()-start)*1000)
   start=time.perf_counter();index=hnswlib.Index(space='cosine',dim=dims);index.init_index(max_elements=count,M=16,ef_construction=200,random_seed=42);index.set_num_threads(1);index.add_items(a);build=time.perf_counter()-start
   for ef in (16,64,128):
    index.set_ef(ef);times_h=[];recall=[]
    for v,target in zip(q,exact):
     start=time.perf_counter();labels,_=index.knn_query(v,k=10);times_h.append((time.perf_counter()-start)*1000);recall.append(len(set(labels[0])&set(target))/10)
    rows.append({'dimensions':dims,'documents':count,'ef_search':ef,'top10_set_recall':float(np.mean(recall)),'hnsw_p50_ms':float(np.median(times_h)),'hnsw_p95_ms':float(np.percentile(times_h,95)),'exact_p50_ms':float(np.median(times)),'build_seconds':build,'vector_bytes':int(a.nbytes)})
   print(dims,count,'complete',flush=True)
 atomic_json('runs/evaluation/index-benchmark.json',{'kind':'Random-vector mechanical test; not semantic relevance','platform':platform.system()+' '+platform.machine(),'query_count':100,'seed':42,'M':16,'ef_construction':200,'threads':1,'rows':rows})
if __name__=='__main__':main()
