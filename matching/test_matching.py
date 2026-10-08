import copy,unittest
from matching.legacy_engine import reciprocal,unpack,rank_candidates,extract
from matching.schema import questions,dimensions,SCHEMA_HASH,attributes

def spec(**changes):
 result={'self':{'hiking':True,'early':False},'prefer':{'hiking':True,'early':False},'require':{'hiking':True,'early':False},'exclude':{'hiking':False,'early':False}}
 for k,v in changes.items():result[k].update(v)
 return result

class MatchingTests(unittest.TestCase):
 def test_one_way_is_not_reciprocal(self):
  a=spec();b=spec(exclude={'early':True,'hiking':True},require={'hiking':False})
  result=reciprocal(a,b);self.assertTrue(result['forward']['eligible']);self.assertFalse(result['reverse']['eligible']);self.assertFalse(result['eligible'])
 def test_unknown_required_evidence_rejects(self):
  a=spec();b=spec(self={'hiking':None});result=reciprocal(a,b)
  self.assertFalse(result['eligible']);self.assertIn('hiking',result['forward']['missing'])
 def test_optional_difference_is_not_hard_rejection(self):
  a=spec(prefer={'early':True});b=spec();result=reciprocal(a,b)
  self.assertTrue(result['eligible']);self.assertLess(result['score'],1)
 def test_contradictory_and_uncertain_requirements_reject(self):
  self.assertFalse(reciprocal(spec(exclude={'hiking':True}),spec())['eligible'])
  self.assertFalse(reciprocal(spec(require={'hiking':None}),spec())['eligible'])
 def test_schema_sizes_and_channel_alignment(self):
  for size in (64,256):
   self.assertTrue(all(__import__('re').fullmatch(r'[A-Za-z0-9_.-]{1,100}',key) for key in questions(size)));self.assertEqual(len(questions(size)),size);self.assertEqual(list(questions(size)),dimensions(size));self.assertEqual(len(attributes(size)),size//4)
 def test_invalid_vector_rejected(self):
  v={'dimensions':64,'schema_hash':SCHEMA_HASH,'dimension_ids':dimensions(64),'values':[.5]*64,'known':[False]*64}
  self.assertEqual(len(unpack(v)['self']),16)
  bad=copy.deepcopy(v);bad['values'][0]=float('nan')
  with self.assertRaises(ValueError):unpack(bad)
  bad=copy.deepcopy(v);bad['known'][0]='false'
  with self.assertRaises(ValueError):unpack(bad)
 def test_extraction_isolates_channel_inputs(self):
  class CaptureClient:
   provider='test';batch_size=64
   def __init__(self):self.calls=[]
   def call(self,state,items):
    self.calls.append((state,items))
    return {'response':{'model':'test','answers':{key:{'probabilities':{'yes':0.,'no':1.},'choice':'no'} for key in items}},'elapsed_seconds':0,'request_hash':'test'}
  state={'about_me':'Own evidence','looking_for':{'prefer':['Optional evidence'],'require':['Required evidence'],'exclude':['Excluded evidence']}}
  for size in (64,256):
   client=CaptureClient();vector=extract(client,state,size);self.assertEqual(len(client.calls),4)
   for channel,(payload,items) in zip(('self','prefer','require','exclude'),client.calls):
    self.assertEqual(payload,{'input':state['about_me'] if channel=='self' else state['looking_for'][channel]})
    self.assertTrue(all(key.startswith(channel+'.') for key in items))
   self.assertEqual(len(vector['values']),size)
  with self.assertRaises(ValueError):extract(CaptureClient(),{'about_me':'x','looking_for':'Unstructured'},64)
 def test_unknown_mass_normalization_stays_in_range(self):
  class Client:
   provider='test';batch_size=64
   def call(self,state,items):
    return {'response':{'model':'test','answers':{key:{'probabilities':{'yes':.07,'no':0.,'mixed':0.,'unknown':.93},'choice':'unknown'} for key in items}},'elapsed_seconds':0,'request_hash':'test'}
  vector=extract(Client(),{'about_me':'','looking_for':{'prefer':[],'require':[],'exclude':[]}},64)
  self.assertTrue(all(x==1. for x in vector['values']))
  self.assertTrue(all(x is None for channel in unpack(vector).values() for x in channel.values()))
 def test_template_control_keeps_unknown_and_conflicting_evidence(self):
  from matching.template_control import parse
  state={'about_me':'I like hills.','looking_for':{'prefer':[],'require':[],'exclude':[]}}
  self.assertIsNone(parse(state)['self']['activity.hiking'])
  state['about_me']='This fits me: enjoys hiking.'
  self.assertIs(parse(state)['self']['activity.hiking'],True)
  state['about_me']+=' This does not fit me: enjoys hiking.'
  self.assertIsNone(parse(state)['self']['activity.hiking'])
 def test_diagnostic_decoder_preserves_explicit_unknown(self):
  from matching.troubleshoot import decode
  vector={'dimensions':64,'schema_hash':SCHEMA_HASH,'dimension_ids':dimensions(64),'values':[.3018]*64,'known':[True]*64,'answers':{key:{'choice':'no'} for key in dimensions(64)}}
  self.assertIsNone(decode(vector,'original')['require']['style.quiet_places'])
  self.assertIs(decode(vector,'condition_choice')['require']['style.quiet_places'],False)
  vector['answers']['self.activity.hiking']['choice']='unknown'
  self.assertIsNone(decode(vector,'selected_choice')['self']['activity.hiking'])
 def test_fixture_pair_integrity_and_no_label_input(self):
  import json
  from pathlib import Path
  from matching.fixtures import build
  data=build();saved=json.loads(Path('matching/data.json').read_text());self.assertEqual(data,saved)
  people={p['id']:p for p in data['people']};self.assertEqual(len(people),164)
  for a,b in data['pairs']:
   self.assertNotEqual(people[a]['state'],people[b]['state']);self.assertTrue(reciprocal(people[a]['truth'],people[b]['truth'])['eligible'])
   self.assertIn(b,data['oracle'][a]['best']);self.assertIn(a,data['oracle'][b]['best'])
   self.assertGreaterEqual(sum(people[a]['truth']['self'][k]!=people[b]['truth']['self'][k] for k in attributes(64)),4)
  for p in people.values():
   self.assertEqual(set(p['state']),{'about_me','looking_for'})
   text=json.dumps(p['state']);self.assertNotIn(p['id'],text)
   if p['kind'] in ('wrong_intent','one_sided','missing_evidence'):
    self.assertFalse(reciprocal(people[p['for_query']]['truth'],p['truth'])['eligible'])
   if p['kind']=='eligible_lower_rank':
    m=reciprocal(people[p['for_query']]['truth'],p['truth']);self.assertTrue(m['eligible']);self.assertLess(m['score'],1)
   if p['kind']=='no_match':self.assertFalse(data['oracle'][p['id']]['eligible'])
if __name__=='__main__':unittest.main()
