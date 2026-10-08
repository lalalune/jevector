import json, math, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from schema import schema, questions, HOBBIES
from jevector import Client, normalize, distance, validate_answers, digest
from benchmark import expected_for, retrieval
from generate_data import build

def answer(q,choice):
 return {'type':'choice','choice':choice,'confidence':1.,'probabilities':{k:float(k==choice) for k in q['criteria']}}

class SchemaTests(unittest.TestCase):
 def test_counts_order_and_groups(self):
  for size,counts in [(64,{'personality':16,'interests':32,'relationships':8,'lifestyle':8}),(256,{'personality':64,'interests':128,'relationships':32,'lifestyle':32})]:
   dims=schema(size);self.assertEqual(len({d['id'] for d in dims}),size)
   self.assertEqual({g:sum(d['group']==g for d in dims) for g in counts},counts)
   self.assertEqual(list(questions(size)),[d['id'] for d in dims])
 def test_all_hobbies_have_positive_and_negative_fixtures(self):
  dataset=build()
  for h in [h for hs in HOBBIES.values() for h in hs]:
   labels={p['expected'].get('interest.'+h) for p in dataset['people']}
   self.assertTrue({'yes','no'}<=labels)
 def test_labels_not_leaked(self):
  for p in build()['people']:
   for state in p['views'].values():
    self.assertNotIn('expected',state);self.assertNotIn('id',state);self.assertNotIn('personality_template',state)
 def test_compact_unknown_is_not_negative(self):
  self.assertNotIn('interest_group.walking_outdoors',expected_for({'interest.hiking':'no'},64))
  self.assertEqual(expected_for({'interest.hiking':'yes'},64)['interest_group.walking_outdoors'],'yes')
 def test_question_order_changes_hash(self):
  self.assertNotEqual(digest({'a':1,'b':2}),digest({'b':2,'a':1}))

class VectorTests(unittest.TestCase):
 def test_unknown_and_negative_distinct(self):
  dims=schema(64);qs=questions(64);unknown={k:answer(q,'unknown') for k,q in qs.items()}
  v=normalize(unknown,dims);self.assertFalse(any(v['mask']));self.assertEqual(distance(v,v,dims),None)
  key=next(d['id'] for d in dims if d['kind']=='interest');unknown[key]=answer(qs[key],'no');v=normalize(unknown,dims);i=[d['id'] for d in dims].index(key)
  self.assertEqual(v['values'][i],0);self.assertTrue(v['mask'][i])
 def test_probability_validation_rejects_nan_and_missing(self):
  q={'x':{'criteria':{'yes':'','no':''}}};a={'x':answer(q['x'],'yes')};validate_answers(a,q)
  a['x']['probabilities']['yes']=float('nan')
  with self.assertRaises(ValueError):validate_answers(a,q)
  with self.assertRaises(ValueError):validate_answers({},q)
 def test_rounded_probabilities_and_provider_choice_preserved(self):
  q={'x':{'criteria':{'yes':'','no':'','unknown':''}}};a={'x':{'type':'choice','choice':'yes','confidence':.2,'probabilities':{'yes':.33,'no':.34,'unknown':.34}}}
  validate_answers(a,q);self.assertEqual(a['x']['choice'],'yes')
 def test_corrupt_probability_sum_rejected(self):
  q={'x':{'criteria':{'yes':'','no':''}}};a={'x':answer(q['x'],'yes')};a['x']['probabilities']['no']=.5
  with self.assertRaises(ValueError):validate_answers(a,q)
 def test_balanced_groups(self):
  dims=schema(64);a={'values':[0]*64,'mask':[True]*64,'evidence':[1]*64};b={**a,'values':[1 if d['group']=='interests' else 0 for d in dims]}
  self.assertAlmostEqual(distance(a,b,dims),.5)
 def test_disjoint_interest_block_is_not_free(self):
  dims=schema(64);query={'values':[.5]*64,'mask':[True]*64,'evidence':[1]*64}
  same={**query,'values':list(query['values'])}
  sparse={**query,'mask':[d['group']!='interests' for d in dims]}
  self.assertEqual(distance(query,same,dims),0)
  self.assertGreater(distance(query,sparse,dims),distance(query,same,dims))
 def test_uncertainty_is_symmetric_and_keeps_unknown_unknown(self):
  dims=schema(64);a={'values':[.5]*64,'mask':[True]*64,'evidence':[1]*64}
  b={**a,'mask':[d['group']=='personality' for d in dims]}
  self.assertEqual(distance(a,b,dims),distance(b,a,dims))
  self.assertEqual(sum(b['mask']),16)
 def test_sparse_candidate_cannot_win_by_hiding_disagreement(self):
  dims=schema(64);a={'values':[.5]*64,'mask':[True]*64,'evidence':[1]*64}
  close={**a,'values':[.55]*64}
  sparse={**a,'mask':[d['group']=='personality' for d in dims]}
  self.assertLess(distance(a,close,dims),distance(a,sparse,dims))
 def test_ties_do_not_get_free_credit(self):
  v={'values':[.5]*64,'mask':[True]*64,'evidence':[1]*64};records={k:v for k in ('x_a','x_b','y_a','y_b')}
  self.assertEqual(retrieval(records,[{'id':'x'},{'id':'y'}],64)['unique_top1'],0)

class ClientTests(unittest.TestCase):
 def test_batch_limits_and_reassembly(self):
  for provider,batch,expected in [('clef',None,4),('jev',None,1),('jev',64,4)]:
   c=Client(provider,account='test',batch_size=batch);calls=[]
   def fake(state,qs,nonce=None):
    calls.append(qs)
    return {'request_hash':str(len(calls)),'elapsed_seconds':0,'response':{'model':'test','usage':{'input_tokens':1},'answers':{k:answer(q,'unknown') for k,q in qs.items()}}}
   with patch.object(c,'call',side_effect=fake):result=c.extract('state',256)
   self.assertEqual(len(calls),expected);self.assertEqual(len(result['values']),256);self.assertEqual(result['dimension_ids'],[d['id'] for d in schema(256)])
  with self.assertRaises(ValueError):Client('clef',account='test',batch_size=256)
 def test_transient_520_retries(self):
  import io,urllib.error
  from unittest.mock import MagicMock
  qs={'x':{'type':'choice','criteria':{'yes':'','no':''}}}
  payload={'model':'test','answers':{'x':answer(qs['x'],'yes')}}
  opener=MagicMock();opener.open.side_effect=[urllib.error.HTTPError('https://example.invalid',520,'temporary',{},None),io.BytesIO(json.dumps(payload).encode())]
  with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.build_opener',return_value=opener),patch('jevector.time.sleep') as sleep:
   result=Client('jev',key='test-only',cache=tmp).call('state',qs)
   self.assertEqual(result['attempts'],2);sleep.assert_called_once()
 def test_disconnect_retry_and_concurrent_request_deduplication(self):
  import io,http.client,threading
  from concurrent.futures import ThreadPoolExecutor
  from unittest.mock import MagicMock
  qs={'x':{'type':'choice','criteria':{'yes':'','no':''}}};payload={'model':'test','answers':{'x':answer(qs['x'],'yes')}}
  opener=MagicMock();opener.open.side_effect=[http.client.RemoteDisconnected(),io.BytesIO(json.dumps(payload).encode())]
  with tempfile.TemporaryDirectory() as tmp,patch('urllib.request.build_opener',return_value=opener),patch('jevector.time.sleep'):
   clients=[Client('jev',key='test-only',cache=tmp) for _ in range(8)];barrier=threading.Barrier(8)
   def work(c):barrier.wait();return c.call('state',qs)
   with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(work,clients))
   self.assertEqual(opener.open.call_count,2);self.assertEqual(len({r['request_hash'] for r in records}),1)
 def test_invalid_network_answer_is_not_cached(self):
  import io
  from unittest.mock import MagicMock
  qs={'x':{'type':'choice','criteria':{'yes':'','no':''}}}
  opener=MagicMock();opener.open.return_value=io.BytesIO(json.dumps({'answers':{}}).encode())
  with tempfile.TemporaryDirectory() as tmp,patch('urllib.request.build_opener',return_value=opener):
   with self.assertRaises(ValueError):Client('jev',key='test-only',cache=tmp).call('state',qs)
   self.assertEqual(list(Path(tmp).rglob('*.json')),[])
 def test_validated_cache_without_credentials(self):
  from jevector import VERSION, atomic_json
  with tempfile.TemporaryDirectory() as tmp:
   c=Client('jev',cache=tmp);qs={'x':{'type':'choice','criteria':{'yes':'','unknown':''}}};body={'model':c.model,'state':'s','questions':qs}
   key=digest({'provider':'jev','account':None,'version':VERSION,'request':body,'nonce':None})
   atomic_json(Path(tmp)/'jev'/(key+'.json'),{'request_hash':key,'response':{'answers':{'x':answer(qs['x'],'yes')}}})
   with patch('urllib.request.build_opener',side_effect=AssertionError('Network should not run')):self.assertEqual(c.call('s',qs)['response']['answers']['x']['choice'],'yes')

if __name__=='__main__':unittest.main()
