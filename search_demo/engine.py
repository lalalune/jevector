"""Separate role encoders, a persisted HNSW index, and dimensional explanations."""
import hashlib,json,math
from pathlib import Path
import hnswlib
import numpy as np
from jevector import Client, atomic_json, digest
from search_demo.schema import VERSION, dimensions, questions

def encode(client,text,size,role):
 qs=list(questions(size,role).items());answers={};records=[]
 for start in range(0,size,client.batch_size):
  record=client.call({'schema':VERSION,'input':text},dict(qs[start:start+client.batch_size]));records.append(record);answers.update(record['response']['answers'])
 # Probability mass is normalized to account for Jev's displayed rounding.
 values=[answers[d['id']]['probabilities']['yes']/sum(answers[d['id']]['probabilities'].values()) for d in dimensions(size)]
 return {'schema_version':VERSION,'role':role,'provider':client.provider,'models':sorted({r['response']['model'] for r in records}),'dimension_ids':[d['id'] for d in dimensions(size)],'values':values,'answers':answers,'input_hash':digest(text),'request_hashes':[r['request_hash'] for r in records],'input_tokens':sum(r['response'].get('usage',{}).get('input_tokens',0) for r in records),'elapsed_seconds':sum(r['elapsed_seconds'] for r in records)}

def matrix(vectors):
 a=np.asarray([v['values'] for v in vectors],dtype=np.float32)
 if a.ndim!=2 or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):raise ValueError('Expected finite 0–1 vector matrix')
 if np.any(np.linalg.norm(a,axis=1)<1e-8):raise ValueError('Cannot index or search a zero vector')
 return a

class SearchIndex:
 def __init__(self,folder):self.folder=Path(folder)
 def build(self,documents,vectors,ef_search=64):
  if not isinstance(ef_search,int) or ef_search<1:raise ValueError('ef_search must be a positive integer')
  if not documents or len(documents)!=len(vectors):raise ValueError('Invalid document/vector count')
  if len({d['id'] for d in documents})!=len(documents):raise ValueError('Duplicate document IDs')
  reference=vectors[0]
  for v in vectors:
   if v['role']!='document' or any(v[k]!=reference[k] for k in ('provider','models','schema_version','dimension_ids')):raise ValueError('Incompatible document vectors')
  a=matrix(vectors);self.folder.mkdir(parents=True,exist_ok=True)
  self.index=hnswlib.Index(space='cosine',dim=a.shape[1]);self.index.init_index(max_elements=len(a),ef_construction=200,M=16,random_seed=42);self.index.set_num_threads(1);self.index.add_items(a,np.arange(len(a)));self.index.set_ef(ef_search)
  self.index.save_index(str(self.folder/'hnsw.bin'));np.save(self.folder/'documents.npy',a,allow_pickle=False)
  self.meta={k:reference[k] for k in ('provider','models','schema_version','dimension_ids')};self.meta.update({'documents':documents,'count':len(a),'metric':'cosine','M':16,'ef_construction':200,'ef_search':ef_search,'hnsw_sha256':hashlib.sha256((self.folder/'hnsw.bin').read_bytes()).hexdigest(),'matrix_sha256':hashlib.sha256((self.folder/'documents.npy').read_bytes()).hexdigest()});atomic_json(self.folder/'manifest.json',self.meta);self.a=a
 def load(self):
  self.meta=json.loads((self.folder/'manifest.json').read_text())
  for filename,key in [('hnsw.bin','hnsw_sha256'),('documents.npy','matrix_sha256')]:
   if hashlib.sha256((self.folder/filename).read_bytes()).hexdigest()!=self.meta[key]:raise ValueError('Index checksum mismatch')
  self.a=np.load(self.folder/'documents.npy',allow_pickle=False)
  self.index=hnswlib.Index(space='cosine',dim=len(self.meta['dimension_ids']));self.index.load_index(str(self.folder/'hnsw.bin'));self.index.set_ef(self.meta['ef_search']);self.index.set_num_threads(1);return self
 def search(self,query,k=5,allow_document_role=False,scoring="cosine",required=None,excluded=None,min_score=None):
  if query['role']!='query' and not allow_document_role:raise ValueError('Search requires a query-role vector')
  if any(query[key]!=self.meta[key] for key in ('provider','models','schema_version','dimension_ids')):raise ValueError('Query and index use incompatible schemas/models')
  k=min(k,self.meta['count'])
  if k<1:raise ValueError('k must be positive')
  q=matrix([query]);eligible=self.eligible(required,excluded)
  if min_score is not None and not 0<=min_score<=1:raise ValueError('min_score must be in [0,1]')
  if scoring not in ('cosine','coverage'):raise ValueError('Unknown scoring method')
  k=min(k,int(eligible.sum()))
  if not k:return []
  if scoring=='coverage':
   # Directional coverage does not penalize document content outside the query.
   # Exact scan: cosine HNSW candidates cannot guarantee coverage recall.
   scores=(self.a@q[0])/q[0].sum();order=np.flatnonzero(eligible)
   order=order[np.argsort(-scores[order],kind='stable')[:k]];labels=[order];dist=[1-scores[order]]
  else:
   self.index.set_ef(max(self.meta['ef_search'],k))
   labels,dist=self.index.knn_query(q,k=k,filter=(lambda label:bool(eligible[label])) if required or excluded else None)
  rows=[];dims=dimensions(q.shape[1]);qn=q[0]/np.linalg.norm(q[0])
  for label,distance in zip(labels[0],dist[0]):
   score=float(1-distance)
   if min_score is not None and score<min_score:continue
   i=int(label);dn=self.a[i]/np.linalg.norm(self.a[i]);contributions=qn*dn
   if scoring=='coverage':contributions=q[0]*self.a[i]/q[0].sum()
   top=np.argsort(-contributions)[:5]
   rows.append({'document_id':self.meta['documents'][i]['id'],'title':self.meta['documents'][i]['title'],'score':score,'scoring':scoring,'cosine_similarity':float(qn@dn),'dimensions':[{'id':dims[j]['id'],'goal':dims[j]['goal'],'facet':dims[j]['facet'],'query_value':float(q[0,j]),'document_value':float(self.a[i,j]),'score_contribution':float(contributions[j]),'cosine_contribution':float(qn[j]*dn[j])} for j in top]})
  return rows
 def eligible(self,required=None,excluded=None):
  allowed=np.ones(self.meta['count'],dtype=bool)
  for constraints,required_mode in ((required or {},True),(excluded or {},False)):
   for key,threshold in constraints.items():
    if key not in self.meta['dimension_ids']:raise ValueError('Unknown constraint dimension: '+key)
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Invalid constraint threshold')
    i=self.meta['dimension_ids'].index(key)
    allowed &= (self.a[:,i]>=threshold) if required_mode else (self.a[:,i]<threshold)
  return allowed
 def exact(self,query,k=5):
  q=matrix([query])[0];scores=(self.a@q)/(np.linalg.norm(self.a,axis=1)*np.linalg.norm(q));order=np.argsort(-scores,kind='stable')[:k]
  return [self.meta['documents'][i]['id'] for i in order]
