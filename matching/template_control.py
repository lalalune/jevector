"""Fixture-language control. No pair IDs, truth attributes, or model outputs are read."""
from matching.schema import ATTRIBUTES
from matching.legacy_engine import reciprocal

def parse(state):
 text=state['about_me'];own={}
 for key,description in ATTRIBUTES.items():
  positive=any(t.format(description) in text for t in ('This fits me: {}.','A description of me is: {}.','About myself: {} is true.'))
  negative=any(t.format(description) in text for t in ('This does not fit me: {}.','I reject this description of myself: {}.','About myself: {} is false.'))
  own[key]=positive if positive!=negative else None
 return {'self':own,**{channel:{key:description in state['looking_for'][channel] for key,description in ATTRIBUTES.items()} for channel in ('prefer','require','exclude')}}

def rankings(states,query_ids):
 decoded={pid:parse(state) for pid,state in states.items()};out={}
 for qid in query_ids:
  candidates=[]
  for pid,state in decoded.items():
   if pid==qid:continue
   match=reciprocal(decoded[qid],state)
   if match['eligible']:candidates.append((pid,match['score']))
  out[qid]=[pid for pid,score in sorted(candidates,key=lambda row:(-row[1],row[0]))]
 return out
