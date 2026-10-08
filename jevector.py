"""Provider-neutral, evidence-aware profile vector extraction (Python 3.11+)."""
import argparse
import datetime
import getpass
import hashlib
import http.client
import json
import math
import os
import tempfile
import threading
import weakref
from pathlib import Path
import time
import urllib.error
import urllib.request
from experiment import token as cloudflare_token
from schema import VERSION, schema, questions

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self, *args, **kwargs):
  raise RuntimeError('Refusing authenticated HTTP redirect')


def digest(value):
 # Order matters to Clef; do not sort question maps when fingerprinting requests.
 return hashlib.sha256(json.dumps(value,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as stream:stream.write(json.dumps(value,indent=2)+'\n')
  os.replace(name,path)
 finally:
  if os.path.exists(name):os.unlink(name)

def validate_answers(answers,qs):
 if set(answers)!=set(qs): raise ValueError('Response question IDs do not match request')
 for key,q in qs.items():
  a=answers[key];p=a.get('probabilities',{})
  if a.get('type')!='choice' or set(p)!=set(q['criteria']): raise ValueError(f'Invalid answer shape: {key}')
  if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in p.values()): raise ValueError(f'Invalid probability: {key}')
  if abs(sum(p.values())-1)>len(p)*.005+.00001: raise ValueError(f'Probabilities do not sum to one: {key}')
  if a.get('choice') not in p: raise ValueError(f'Invalid selected choice: {key}')
  c=a.get('confidence')
  if not isinstance(c,(int,float)) or not math.isfinite(c) or not 0<=c<=1: raise ValueError(f'Invalid confidence: {key}')

def normalize(answers,dims):
 values=[];evidence=[];mask=[]
 for d in dims:
  a=answers[d['id']];p=a['probabilities'];known=(sum(p.values())-p['unknown'])/sum(p.values())
  weights={'low':0,'somewhat_low':.25,'mixed':.5,'somewhat_high':.75,'high':1} if d['kind']=='trait' else {'no':0,'mixed':.5,'yes':1}
  values.append(sum(p[k]*w for k,w in weights.items())/sum(p[k] for k in weights) if sum(p[k] for k in weights)>1e-9 else .5)
  evidence.append(known)
  # Chosen unknown is not a measured midpoint. Mixed preference is evidence but not a yes.
  mask.append(known>=.6 and a['choice']!='unknown')
 return {'values':values,'evidence':evidence,'mask':mask}

class Client:
 _cache_locks=weakref.WeakValueDictionary()
 _cache_guard=threading.Lock()
 def __init__(self,provider,account=None,model=None,key=None,cache='runs/cache',batch_size=None):
  if provider not in ('clef','jev'): raise ValueError('Unknown provider')
  self.provider=provider;self.model=model or ('clef-flash' if provider=='clef' else 'jev-1.13.0')
  self.account=account or os.environ.get('CLOUDFLARE_ACCOUNT_ID');self.key=key
  self.cache=Path(cache);self.batch_size=batch_size or (64 if provider=='clef' else 256)
  if not 1<=self.batch_size<=(64 if provider=='clef' else 256): raise ValueError('Invalid batch size')
  if provider=='clef' and (not self.account or self.model not in ('clef','clef-flash')): raise ValueError('Clef needs an account and model clef or clef-flash')
 def call(self,state,qs,nonce=None):
  lock_key=(str(self.cache.resolve()),self.provider,self.model,self.account,digest({'state':state,'questions':qs,'nonce':nonce}))
  with self._cache_guard:lock=self._cache_locks.setdefault(lock_key,threading.RLock())
  with lock:return self._call(state,qs,nonce)
 def _call(self,state,qs,nonce=None):
  body={'model':self.model,'state':state,'questions':qs}
  identity={'provider':self.provider,'account':self.account if self.provider=='clef' else None,'version':VERSION,'request':body,'nonce':nonce}
  request_hash=digest(identity);path=self.cache/self.provider/(request_hash+'.json')
  if path.exists():
   record=json.loads(path.read_text())
   if record['request_hash']!=request_hash: raise ValueError('Cache fingerprint mismatch')
   validate_answers(record['response']['answers'],qs)
   return record
  secret=(self.key or os.environ.get('TYPESAFE_API_KEY') or os.environ.get('JEV_API_KEY')) if self.provider=='jev' else cloudflare_token()
  if not secret: raise ValueError('Set TYPESAFE_API_KEY or use --prompt-key')
  url='https://api.typesafe.ai/v1/systemone' if self.provider=='jev' else f'https://api.cloudflare.com/client/v4/accounts/{self.account}/ai/run/@cf/cloudflare/{self.model}'
  req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
  start=time.monotonic()
  for attempt in range(4):
   try:
    with urllib.request.build_opener(NoRedirect).open(req,timeout=120) as r:
     data=json.load(r)
    break
   except urllib.error.HTTPError as e:
    if e.code in (429,500,502,503,504,520,521,522,524,529) and attempt<3:
     retry=e.headers.get('Retry-After','')
     time.sleep(min(float(retry),30) if retry.isdigit() else 2**attempt);continue
    # Do not expose response bodies, which may contain input or authentication details.
    raise RuntimeError(f'{self.provider} HTTP {e.code}; request {request_hash[:12]}') from None
   except (urllib.error.URLError,http.client.RemoteDisconnected,TimeoutError,ConnectionError):
    if attempt==3:raise RuntimeError(f'{self.provider} connection failed; request {request_hash[:12]}') from None
    time.sleep(2**attempt)
  if self.provider=='clef':
   if not data.get('success'): raise RuntimeError('Cloudflare reported an unsuccessful request')
   data=data['result']
  record={'request_hash':request_hash,'provider':self.provider,'requested_model':self.model,'schema_version':VERSION,'question_ids':list(qs),'question_count':len(qs),'state_hash':digest(state),'elapsed_seconds':time.monotonic()-start,'attempts':attempt+1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'response':data}
  validate_answers(data['answers'],qs)
  atomic_json(path,record)
  return record
 def extract(self,state,size=256,nonce=None):
  dims=schema(size);items=list(questions(size).items());answers={};records=[]
  for i in range(0,len(items),self.batch_size):
   record=self.call(state,dict(items[i:i+self.batch_size]),nonce=nonce);records.append(record);answers.update(record['response']['answers'])
  return {'schema_version':VERSION,'provider':self.provider,'models':sorted({r['response']['model'] for r in records}),'dimension_ids':[d['id'] for d in dims],**normalize(answers,dims),'answers':answers,'diagnostics':{'choice_argmax_disagreements':[k for k,a in answers.items() if a['probabilities'][a['choice']]<max(a['probabilities'].values())-.0002],'probability_sum_deviations':{k:sum(a['probabilities'].values())-1 for k,a in answers.items() if abs(sum(a['probabilities'].values())-1)>.002}},'request_hashes':[r['request_hash'] for r in records],'elapsed_seconds':sum(r['elapsed_seconds'] for r in records),'input_tokens':sum(r['response'].get('usage',{}).get('input_tokens',0) for r in records)}

def legacy_distance(a,b,dims,groups=None,min_shared=8):
 if len(a['values'])!=len(dims) or len(b['values'])!=len(dims): raise ValueError('Dimension mismatch')
 buckets={};shared=0
 for i,d in enumerate(dims):
  if groups and d['group'] not in groups: continue
  if not(a['mask'][i] and b['mask'][i]):continue
  shared+=1;w=min(a['evidence'][i],b['evidence'][i]);bucket=buckets.setdefault(d['group'],[0.,0.]);bucket[0]+=w*(a['values'][i]-b['values'][i])**2;bucket[1]+=w
 if shared<min_shared:return None
 return math.sqrt(sum(s/w for s,w in buckets.values())/len(buckets))

MATCHER_VERSION = 'coverage-v1'

def distance(a,b,dims,groups=None,min_shared=8):
 """Balanced observed error plus uncertainty for unshared evidence.

 Missingness is not a negative preference: .25 is an uncertainty cost (the
 squared distance from a binary endpoint to its midpoint), never an inferred
 rejection. Blocks with evidence on either side participate in the denominator.
 """
 if len(a['values'])!=len(dims) or len(b['values'])!=len(dims):
  raise ValueError('Dimension mismatch')
 buckets={};shared=0
 for i,d in enumerate(dims):
  if groups and d['group'] not in groups: continue
  am,bm=a['mask'][i],b['mask'][i]
  if not(am or bm):continue
  bucket=buckets.setdefault(d['group'],[0.,0.,0.])
  aw=a['evidence'][i] if am else 0.
  bw=b['evidence'][i] if bm else 0.
  bucket[2]+=max(aw,bw)
  if am and bm:
   shared+=1;w=min(aw,bw)
   bucket[0]+=w*(a['values'][i]-b['values'][i])**2
   bucket[1]+=w
 if shared<min_shared:return None
 costs=[]
 for error,overlap,union in buckets.values():
  observed=error/overlap if overlap else 0.
  uncertainty=.25*(1-overlap/union)
  costs.append(observed+uncertainty)
 return math.sqrt(sum(costs)/len(costs))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--provider',choices=['clef','jev'],required=True);p.add_argument('--dimensions',type=int,choices=[64,256],default=256);p.add_argument('--account');p.add_argument('--model');p.add_argument('--batch-size',type=int);p.add_argument('--prompt-key',action='store_true');p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 key=getpass.getpass('Jev API key (hidden): ') if args.prompt_key else None
 state=json.loads(args.input.read_text()) if args.input.suffix=='.json' else args.input.read_text()
 result=Client(args.provider,args.account,args.model,key,batch_size=args.batch_size).extract(state,args.dimensions);atomic_json(args.output,result)
 print(f'Saved {len(result["values"])} dimensions; {sum(result["mask"])} supported by evidence; models {result["models"]}')
if __name__=='__main__':main()
