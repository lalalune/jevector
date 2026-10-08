"""HNSW candidate retrieval over cross-role decision-model projections."""
import copy
import numpy as np
import hnswlib
from matching.engine import prepare,rank_decoded

RETRIEVAL_POLICY={'projection':'cross-role-1','space':'inner-product','M':16,'construction_ef':200}

def projections(v):
 n=v['dimensions']//4
 # Own attributes and requested partner attributes occupy different roles.
 raw=np.asarray(v['values'],dtype=np.float32).reshape(4,n)
 own=raw[0]*np.asarray(v['known'][:n],dtype=np.float32)
 # Soft preference evidence plus required-positive and excluded-negative evidence.
 wanted=raw[1]+raw[2]-raw[3]
 wanted/=max(float(np.abs(wanted).sum()),1.)
 zeros=np.zeros(n,dtype=np.float32)
 document=np.concatenate((own,wanted,zeros,zeros))
 query=np.concatenate((wanted,own,zeros,zeros))
 return document,query

class ProfileIndex:
 def __init__(self,vectors,ef=64):
  if not vectors:raise ValueError('Cannot index an empty gallery')
  if type(ef) is not int or ef<1:raise ValueError('ef must be positive')
  self.decoded=copy.deepcopy(prepare(vectors));self.ids=sorted(vectors);self.positions={k:i for i,k in enumerate(self.ids)};self.ef=ef
  self.documents=np.asarray([projections(vectors[k])[0] for k in self.ids],dtype=np.float32)
  self.query_vectors=np.asarray([projections(vectors[k])[1] for k in self.ids],dtype=np.float32)
  self.index=hnswlib.Index(space='ip',dim=self.documents.shape[1]);self.index.init_index(max_elements=len(self.ids),ef_construction=200,M=16,random_seed=17)
  self.index.add_items(self.documents,np.arange(len(self.ids)),num_threads=1);self.index.set_num_threads(1)
 def candidates(self,query_id,limit=32):
  if type(limit) is not int or limit<1:raise ValueError('candidate limit must be positive')
  if query_id not in self.positions:raise ValueError('Unknown query ID')
  count=min(len(self.ids),limit+1);self.index.set_ef(max(self.ef,count))
  labels,distances=self.index.knn_query(self.query_vectors[self.positions[query_id]],k=count)
  return [self.ids[int(i)] for i in labels[0] if self.ids[int(i)]!=query_id][:limit]
 def search(self,query_id,limit=32,k=5,exact_fallback=False):
  ids=self.candidates(query_id,limit);rows=rank_decoded(self.decoded,query_id,ids);fallback=False
  if exact_fallback and len(rows)<k:
   rows=rank_decoded(self.decoded,query_id);fallback=True
  return {'results':rows[:k],'candidate_count':len(ids),'candidate_ids':ids,'exact_fallback_used':fallback,'exhaustive':fallback or len(ids)==len(self.ids)-1,'retrieval_policy':RETRIEVAL_POLICY,'ef':max(self.ef,min(len(self.ids),limit+1))}
