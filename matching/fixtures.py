"""Generate distinct people from a reciprocal match specification; never from pair labels in prompts."""
import copy,json,random
from pathlib import Path
from jevector import atomic_json,digest
from matching.schema import CORE,EXTRA,ATTRIBUTES,VERSION,SCHEMA_HASH
from matching.legacy_engine import reciprocal
SEED=73021

def preferences(rng,partner):
 base=[k for k in CORE if partner[k] is True];other=[k for k in EXTRA if partner[k] is True]
 intent=next(k for k in base if k.startswith('intent.'))
 required=[intent]+rng.sample([k for k in base if not k.startswith('intent.')],2)
 excluded=rng.sample([k for k in CORE if partner[k] is False and not k.startswith('intent.')],2)
 prefer=rng.sample(base,min(4,len(base)))+rng.sample(other,min(4,len(other)))
 return {'prefer':{k:k in prefer for k in ATTRIBUTES},'require':{k:k in required for k in ATTRIBUTES},'exclude':{k:k in excluded for k in ATTRIBUTES}}

def own(rng,intent):
 result={k:None for k in ATTRIBUTES}
 for k in CORE:result[k]=(k==intent) if k.startswith('intent.') else bool(rng.getrandbits(1))
 for k in rng.sample(list(EXTRA),12):result[k]=bool(rng.getrandbits(1))
 return result

def describe(person,index,rng):
 # Each biography and phrasing are generated separately; no pair or candidate identity is included.
 jobs=['repair technician','library assistant','delivery planner','ceramics tutor','bookkeeper','sound technician','museum assistant','software tester']
 openings=[f'I work as a {jobs[index%len(jobs)]}. I am an adult.',f'An adult working as a {jobs[index%len(jobs)]}.',f'My work is in a {jobs[index%len(jobs)]} role. I am over 21.']
 yes=['This fits me: {}.','A description of me is: {}.','About myself: {} is true.']
 no=['This does not fit me: {}.','I reject this description of myself: {}.','About myself: {} is false.']
 items=list(person['truth']['self'].items());rng.shuffle(items)
 sentences=[openings[index%3]]
 for key,value in items:
  if value is not None:sentences.append((yes if value else no)[index%3].format(ATTRIBUTES[key]))
 conditions={}
 for role in ('prefer','require','exclude'):
  statements=[ATTRIBUTES[key] for key,value in person['truth'][role].items() if value]
  rng.shuffle(statements);conditions[role]=statements
 return {'about_me':' '.join(sentences),'looking_for':conditions}


def build():
 rng=random.Random(SEED);people=[];pairs=[];intents=[k for k in CORE if k.startswith('intent.')]
 def identifier():return 'p_'+''.join(rng.choices('abcdef0123456789',k=12))
 for i in range(16):
  intent=intents[i%4]
  while True:
   a=own(rng,intent);b=own(rng,intent)
   shared=[k for k in CORE if k.startswith('activity.') and a[k] and b[k]]
   differences=sum(a[k]!=b[k] for k in CORE)
   if shared and differences>=4:break
  left={'id':identifier(),'kind':'paired','truth':{'self':a,**preferences(rng,b)}}
  right={'id':identifier(),'kind':'paired','truth':{'self':b,**preferences(rng,a)}}
  left['partner']=right['id'];right['partner']=left['id'];people +=[left,right];pairs.append([left['id'],right['id']])
 anchors=list(people)
 for requester in anchors:
  target=next(p for p in anchors if p['id']==requester['partner'])
  for kind in ('wrong_intent','one_sided','missing_evidence'):
   decoy=copy.deepcopy(target);decoy.update({'id':identifier(),'kind':kind,'for_query':requester['id']});decoy.pop('partner',None)
   # Preserve the shared activities. Change only the condition that makes this a hard negative.
   if kind=='wrong_intent':
    current=next(k for k in intents if decoy['truth']['self'][k]);new=next(k for k in intents if k!=current)
    for k in intents:decoy['truth']['self'][k]=(k==new)
   elif kind=='one_sided':
    key=next(k for k in CORE if not k.startswith('intent.') and requester['truth']['self'][k] is True)
    decoy['truth']['require'][key]=False;decoy['truth']['exclude'][key]=True
   else:
    key=next(k for k,v in requester['truth']['require'].items() if v and not k.startswith('intent.'))
    decoy['truth']['self'][key]=None
   people.append(decoy)
 for i in range(4):
  state=own(rng,intents[i]);requirements=preferences(rng,state)
  # No candidate offers two different connection intentions in this fixture.
  requirements['require'][intents[0]]=True;requirements['require'][intents[1]]=True
  requirements['exclude'][intents[0]]=False;requirements['exclude'][intents[1]]=False
  people.append({'id':identifier(),'kind':'no_match','truth':{'self':state,**requirements}})
 for i,person in enumerate(people):person['state']=describe(person,i,rng)
 # Valid alternatives exercise preference ranking after reciprocal hard checks pass.
 for requester in anchors:
  target=next(p for p in anchors if p['id']==requester['partner'])
  decoy=copy.deepcopy(target);decoy.update({'id':identifier(),'kind':'eligible_lower_rank','for_query':requester['id']});decoy.pop('partner',None)
  candidates=[k for k,v in requester['truth']['prefer'].items() if v and not requester['truth']['require'][k] and not requester['truth']['exclude'][k]]
  candidates.sort(key=lambda k:(k not in CORE,k))
  assert candidates
  decoy['truth']['self'][candidates[0]]=False
  assert reciprocal(requester['truth'],decoy['truth'])['eligible']
  assert reciprocal(requester['truth'],decoy['truth'])['score']<1
  decoy['state']=describe(decoy,len(people),rng);people.append(decoy)
 rng.shuffle(people)
 queries=[p['id'] for p in people if p['kind'] in ('paired','no_match')]
 oracle={}
 for qid in queries:
  q=next(p for p in people if p['id']==qid);ranked=[]
  for p in people:
   if p['id']==qid:continue
   result=reciprocal(q['truth'],p['truth'])
   if result['eligible']:ranked.append({'id':p['id'],'score':result['score']})
  ranked.sort(key=lambda x:(-x['score'],x['id']))
  oracle[qid]={'eligible':[r['id'] for r in ranked],'best':[r['id'] for r in ranked if abs(r['score']-ranked[0]['score'])<1e-9] if ranked else [],'ranking':ranked}
  if q['kind']=='paired':
   assert q['partner'] in oracle[qid]['best']
   assert q['state']!=next(p for p in people if p['id']==q['partner'])['state']
  else:assert not ranked
 return {'version':VERSION,'schema_hash':SCHEMA_HASH,'seed':SEED,'provenance':'Deterministic synthetic constraint-first generation. Distinct paired people; three hard distractors and one eligible lower-ranked alternative per paired query. Not human compatibility labels.','people':people,'pairs':pairs,'query_ids':queries,'oracle':oracle}

def main():
 data=build();atomic_json('matching/data.json',data)
 # Only these states go to models. Pair labels and truth stay in the evaluator.
 atomic_json('matching/example-profile.json',next(p['state'] for p in data['people'] if p['kind']=='paired'))
 atomic_json('matching/model-inputs.json',{p['id']:p['state'] for p in data['people']})
 atomic_json('matching/protocol.json',{'version':VERSION,'schema_hash':SCHEMA_HASH,'seed':SEED,'data_hash':digest(data),'people':len(data['people']),'pairs':len(data['pairs']),'query_count':len(data['query_ids']),'self_thresholds':[.3,.7],'required_known_mass':.6,'ranking':'Both directions must satisfy hard requirements; mean directional optional-preference coverage. Unknown required evidence rejects.','primary_metrics':['designated partner top1','designated partner recall5','oracle best top1','ground-truth hard violations','abstentions','both-direction pair top1'],'tuning':'Development fixture. Prompts and input contract revised after pilot errors. Thresholds fixed; no held-out performance claim.'})
 print('Generated',len(data['people']),'distinct profiles,',len(data['pairs']),'pairs and',len(data['query_ids']),'queries.')
if __name__=='__main__':main()
