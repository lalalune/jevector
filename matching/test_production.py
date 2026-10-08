import copy,json,tempfile,threading,unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from jevector import atomic_json,digest
from matching.schema import attributes,questions
from matching.engine import extract,from_answers,contract,reciprocal,direction,rank_pool,unpack,POLICY_HASH,seal
from matching.ann import ProfileIndex
from matching.cli import load_gallery
from matching.rules import validate,evaluate
from matching.reference import match


def vector(pid='a',own=None,prefer=None,require=None,exclude=None,facts=None,rules=None,size=64):
 state={'profile_id':pid,'about_me':'test','looking_for':{'prefer':prefer or [],'require':require or [],'exclude':exclude or []},'facts':facts or {},'rules':rules or []};answers={}
 for key,q in questions(size).items():
  channel,attribute=key.split('.',1)
  chosen=('unknown' if own is None or own.get(attribute) is None else 'yes' if own[attribute] else 'no') if channel=='self' else ('yes' if attribute in state['looking_for'][channel] else 'no')
  answers[key]={'type':'choice','choice':chosen,'confidence':1.,'probabilities':{k:float(k==chosen) for k in q['criteria']}}
 return from_answers(state,size,answers,'test',['test-model'])

class ProductionTests(unittest.TestCase):
 def test_unsupported_hard_condition_rejected_before_encoding(self):
  with self.assertRaisesRegex(ValueError,'Unsupported require'):vector(require=['activity.cycling'])
  with self.assertRaisesRegex(ValueError,'Unsupported exclude'):vector(exclude=['unknown attribute'])
  class NoNetwork:
   def call(self,*args):raise AssertionError('Unsupported requirements must fail before API calls')
  with self.assertRaisesRegex(ValueError,'Unsupported require'):extract(NoNetwork(),{'profile_id':'x','about_me':'','looking_for':{'prefer':[],'require':['activity.cycling'],'exclude':[]}},64)
 def test_missing_optional_keeps_denominator_and_no_empty_bonus(self):
  a=unpack(vector('a',{'activity.hiking':True},prefer=['activity.hiking','activity.cooking']));b=unpack(vector('b',{'activity.hiking':True,'activity.cooking':False}))
  before=reciprocal(a,b)['score'];a['prefer']['activity.cooking']=None;self.assertEqual(reciprocal(a,b)['score'],before)
  self.assertFalse(reciprocal(unpack(vector('x')),unpack(vector('y')))['eligible'])
  self.assertEqual(reciprocal(unpack(vector('x')),unpack(vector('y')))['score'],0)
  c=unpack(vector('c',{'activity.hiking':True},prefer=['activity.hiking','activity.cycling']))
  self.assertEqual(direction(c,b)['score'],.5);self.assertEqual(c['unresolved_preference_count'],1)
 def test_no_probability_inflation(self):
  v=vector('a',{'activity.hiking':True});a=v['answers']['self.activity.hiking'];a['probabilities']={'yes':.5893,'no':.1055,'mixed':.0848,'unknown':.2204}
  state={'profile_id':'a','about_me':'','looking_for':{'prefer':[],'require':[],'exclude':[]}}
  rebuilt=from_answers(state,64,v['answers'],'test',['test-model']);self.assertIsNone(unpack(rebuilt)['self']['activity.hiking'])
 def test_stable_identity_and_collision_handling(self):
  a=vector('a',{'activity.hiking':True},prefer=['activity.hiking']);b=vector('b',{'activity.hiking':True},prefer=['activity.hiking'])
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'a').mkdir();(root/'b').mkdir();qa=root/'a'/'person.json';gb=root/'b'/'person.json';atomic_json(qa,a);atomic_json(gb,b)
   q,vs=load_gallery(qa,[gb,qa]);self.assertEqual(set(vs),{'a','b'});self.assertEqual(rank_pool(vs,[q])[q][0]['id'],'b')
   clone=root/'copy.json';atomic_json(clone,a);self.assertEqual(len(load_gallery(qa,[clone])[1]),1)
   changed=copy.deepcopy(a);changed['input_hash']='changed';seal(changed);atomic_json(clone,changed)
   with self.assertRaises(ValueError):load_gallery(qa,[clone])
 def test_provenance_policy_and_tamper_checks(self):
  v=vector();bad=copy.deepcopy(v);bad.pop('provider')
  with self.assertRaises(ValueError):unpack(bad)
  bad=copy.deepcopy(v);bad['policy_hash']='old';seal(bad)
  with self.assertRaises(ValueError):unpack(bad)
  bad=copy.deepcopy(v);bad['values'][0]=1
  with self.assertRaises(ValueError):unpack(bad)
 def test_concurrent_atomic_writes(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'same.json';barrier=threading.Barrier(8)
   def work(i):barrier.wait();atomic_json(path,{'value':i,'body':'x'*10000})
   with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(work,range(8)))
   self.assertIn(json.loads(path.read_text())['value'],range(8));self.assertEqual(list(Path(tmp).glob('*.tmp')),[])
 def test_rule_alternatives_ranges_distances_and_unknown(self):
  own={'activity.hiking':False};candidate={'activity.hiking':False,'activity.cooking':True}
  alternatives={'any':[{'attribute':'activity.hiking','value':True},{'attribute':'activity.cooking','value':True}]}
  validate(alternatives,64);self.assertIs(evaluate(alternatives,own,candidate,{},{}),True)
  self.assertIsNone(evaluate({'field':'budget','max':20},own,candidate,{},{}))
  self.assertIs(evaluate({'field':'budget','min':10,'max':20},own,candidate,{}, {'budget':15}),True)
  self.assertIs(evaluate({'distance_km':1},own,candidate,{'location':[0,0]},{'location':[0,1]}),False)
  self.assertIsNone(evaluate({'if':{'attribute':'activity.hiking','value':True},'then':{'field':'budget','max':20}},own,{}, {},{}))
  with self.assertRaises(ValueError):validate({'field':'budget','min':20,'max':10},64)
 def test_independent_reference_and_symmetry(self):
  import random
  rng=random.Random(83);keys=list(attributes(64))
  for i in range(100):
   decoded=[]
   for side in ('a','b'):
    own={k:rng.choice([True,False,None]) for k in keys};prefs=rng.sample(keys,3);required=rng.sample(keys,2)
    decoded.append(unpack(vector(side,own,prefs,required)))
   expected=match(*decoded);actual=reciprocal(*decoded);self.assertEqual(actual['eligible'],expected['eligible']);self.assertEqual(actual['score'],expected['score']);self.assertEqual(actual['score'],reciprocal(*reversed(decoded))['score'])
 def test_hnsw_candidate_boundary_and_full_pool_agreement(self):
  vs={f'p{i}':vector(f'p{i}',{'activity.hiking':i%2==0,'activity.cooking':i%2!=0},['activity.hiking'],['activity.hiking']) for i in range(24)}
  index=ProfileIndex(vs);limited=index.search('p0',limit=3,k=10)
  self.assertLessEqual(limited['candidate_count'],3);self.assertTrue(all(r['id'] in limited['candidate_ids'] for r in limited['results']));self.assertFalse(limited['exhaustive'])
  exact=rank_pool(vs,['p0'])['p0'];all_rows=index.search('p0',limit=24,k=24);self.assertEqual(all_rows['results'],exact);self.assertTrue(all_rows['exhaustive'])
  fallback=index.search('p0',limit=1,k=10,exact_fallback=True);self.assertTrue(fallback['exact_fallback_used']);self.assertEqual(fallback['results'],exact[:10])
 def test_qualification_families_and_extra_dimension_signal(self):
  from matching.qualification_data import build
  d=build();byid={p['id']:p for p in d['people']};self.assertTrue({p['family'] for p in d['people'] if p['split']=='development'}.isdisjoint({p['family'] for p in d['people'] if p['split']=='validation'}))
  for a,b in d['pairs']:
   alt=byid[b+'-alternative'];self.assertTrue(match(byid[a]['truth'],alt['truth'])['eligible']);self.assertLess(match(byid[a]['truth'],alt['truth'])['score'],match(byid[a]['truth'],byid[b]['truth'])['score'])
if __name__=='__main__':unittest.main()
