# Historical benchmark implementation. Use matching.engine for new work.
"""Direction-specific preferences and reciprocal, fail-closed constraints."""
import math
from matching.schema import VERSION,SCHEMA_HASH,attributes,questions,dimensions

def extract(client,state,size):
 if set(state)!={'about_me','looking_for'} or not isinstance(state['about_me'],str) or not isinstance(state['looking_for'],dict) or set(state['looking_for'])!={'prefer','require','exclude'}:raise ValueError('Expected about_me text and prefer/require/exclude lists in looking_for')
 if any(not isinstance(items,list) or not all(isinstance(x,str) for x in items) for items in state['looking_for'].values()):raise ValueError('Partner conditions must be text lists')
 answers={};records=[];all_questions=questions(size)
 # Separate model contexts prevent own traits and partner-condition types from leaking.
 for channel in ('self','prefer','require','exclude'):
  items=[(key,value) for key,value in all_questions.items() if key.startswith(channel+'.')]
  text=state['about_me'] if channel=='self' else state['looking_for'][channel]
  for i in range(0,len(items),client.batch_size):
   r=client.call({'input':text},dict(items[i:i+client.batch_size]));answers.update(r['response']['answers']);records.append(r)
 values=[];known=[]
 for key in dimensions(size):
  a=answers[key];p=a['probabilities'];mass=sum(p.values());observed=sum(value for outcome,value in p.items() if outcome!='unknown')
  values.append((p['yes']+.5*p.get('mixed',0))/observed if observed>1e-9 else .5)
  known.append(observed/mass>=.6 and a['choice'] not in ('unknown','mixed'))
 return {'schema_version':VERSION,'schema_hash':SCHEMA_HASH,'dimensions':size,'provider':client.provider,'models':sorted({r['response']['model'] for r in records}),'dimension_ids':dimensions(size),'values':values,'known':known,'answers':answers,'input_tokens':sum(r['response'].get('usage',{}).get('input_tokens',0) for r in records),'request_seconds':sum(r['elapsed_seconds'] for r in records),'request_hashes':[r['request_hash'] for r in records]}

def unpack(vector):
 size=vector['dimensions'];keys=list(attributes(size));n=len(keys)
 if vector['schema_hash']!=SCHEMA_HASH or vector['dimension_ids']!=dimensions(size):raise ValueError('Incompatible matching schema')
 if len(vector['values'])!=size or len(vector['known'])!=size or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in vector['values']) or any(type(x) is not bool for x in vector['known']):raise ValueError('Invalid matching vector')
 result={channel:{} for channel in ('self','prefer','require','exclude')}
 for j,channel in enumerate(result):
  for i,key in enumerate(keys):
   index=j*n+i;value=vector['values'][index]
   result[channel][key]=(True if value>=.7 else False if value<=.3 else None) if vector['known'][index] else None
 return result

def direction(requester,candidate):
 violations=[];missing=[]
 for key in requester['self']:
  actual=candidate['self'].get(key)
  required=requester['require'].get(key);excluded=requester['exclude'].get(key)
  if required is None or excluded is None:missing.append('uncertain requirement: '+key);continue
  if required and excluded:violations.append('contradictory requirement: '+key)
  elif required or excluded:
   if actual is None:missing.append(key)
   elif required!=actual:violations.append(key)
 preferred=[k for k,v in requester['prefer'].items() if v is True]
 score=sum(candidate['self'].get(k) is True for k in preferred)/len(preferred) if preferred else 1.
 return {'eligible':not violations and not missing,'score':score,'violations':violations,'missing':missing}

def reciprocal(a,b):
 ab=direction(a,b);ba=direction(b,a)
 return {'eligible':ab['eligible'] and ba['eligible'],'score':(ab['score']+ba['score'])/2,'forward':ab,'reverse':ba}

def rank_pool(vectors,query_ids):
 if not vectors:return {key:[] for key in query_ids}
 reference=next(iter(vectors.values()))
 for v in vectors.values():
  if any(v.get(k)!=reference.get(k) for k in ('schema_hash','dimensions','provider','models')):raise ValueError('Mixed provider, model, or schema')
 decoded={key:unpack(v) for key,v in vectors.items()};result={}
 for query_id in query_ids:
  out=[]
  for key,profile in decoded.items():
   if key==query_id:continue
   row=reciprocal(decoded[query_id],profile)
   if row['eligible']:out.append({'id':key,**row})
  result[query_id]=sorted(out,key=lambda row:(-row['score'],row['id']))
 return result

def rank_candidates(query_id,vectors):return rank_pool(vectors,[query_id])[query_id]
