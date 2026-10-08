"""Independent declarative reference; never imports the production matcher."""
import math

def rule_truth(rule,requester,candidate):
 def possibilities(node):
  if 'attribute' in node:
   value=candidate['self'].get(node['attribute'])
   return {False,True} if value is None else {value==node['value']}
  if 'any' in node or 'all' in node:
   import itertools
   group='any' if 'any' in node else 'all';operator=any if group=='any' else all
   return {operator(values) for values in itertools.product(*(possibilities(child) for child in node[group]))}
  if 'if' in node:
   condition=possibilities(node['if'])
   return {False,True} if len(condition)>1 else {True} if condition=={False} else possibilities(node['then'])
  if 'field' in node:
   value=candidate.get('facts',{}).get(node['field'])
   return {False,True} if value is None else {node.get('min',-math.inf)<=value<=node.get('max',math.inf)}
  a=requester.get('facts',{}).get('location');b=candidate.get('facts',{}).get('location')
  if a is None or b is None:return {False,True}
  def xyz(p):
   lat,lon=map(math.radians,p);return (math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat))
  distance=6371.0088*math.acos(max(-1,min(1,sum(x*y for x,y in zip(xyz(a),xyz(b))))))
  return {distance<=node['distance_km']+1e-7}
 return possibilities(rule)=={True}

def match(a,b):
 valid=True;scores=[]
 for requester,candidate in ((a,b),(b,a)):
  positives={k for k,v in requester['require'].items() if v};negatives={k for k,v in requester['exclude'].items() if v}
  valid &= not positives.intersection(negatives)
  valid &= all(candidate['self'].get(k) is True for k in positives)
  valid &= all(candidate['self'].get(k) is False for k in negatives)
  valid &= all(rule_truth(r,requester,candidate) for r in requester.get('rules',[]))
  wants={k for k,v in requester['prefer'].items() if v}
  if wants:scores.append(len({k for k in wants if candidate['self'].get(k) is True})/len(wants))
 valid &= any(v is not None for v in a['self'].values()) and any(v is not None for v in b['self'].values())
 signal=scores or any(a['require'].values()) or any(b['require'].values()) or any(a['exclude'].values()) or any(b['exclude'].values()) or a.get('rules') or b.get('rules')
 return {'eligible':bool(valid and signal),'score':sum(scores)/len(scores) if scores else 0.}
