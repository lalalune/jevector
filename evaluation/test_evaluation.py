import unittest
import numpy as np
from evaluation.common import chunks,plain,rrf,metrics
from evaluation.lexical import BM25
from search_demo.run import bm25
from evaluation.public_run import evaluate
class EvaluationTests(unittest.TestCase):
 def test_bm25_matches_reference(self):
  texts=['cats and dogs','only dogs','birds'];docs=[{'id':str(i),'title':'','body':t} for i,t in enumerate(texts)]
  scorer=BM25(texts)
  for q in ['cats','dogs dogs','no match']:
   ranked=[str(i) for i in np.argsort(-scorer.score(q),kind='stable')]
   self.assertEqual(ranked,bm25(docs,q))
 def test_baseline_audit_scalar_and_signed_rendering(self):
  from matching.baseline_audit import scalar_bm25,natural_preferences,natural_self
  docs=['hiking hiking quiet','cooking quiet'];queries=['hiking hiking','cooking','absent']
  np.testing.assert_allclose(scalar_bm25(docs,queries),np.asarray([BM25(docs).score(q) for q in queries]),rtol=1e-12)
  text=natural_preferences({'looking_for':{'prefer':['enjoys cooking'],'require':['seeks friendship'],'exclude':['enjoys hiking']}})
  self.assertIn('must not be someone who enjoys hiking',text)
  self.assertEqual(natural_self('About myself: enjoys hiking is false.'),'It is false that this person enjoys hiking.')
 def test_chunks_retain_all_words(self):
  import re
  class Tokenizer:
   def encode(self,text,add_special_tokens=True):
    spans=[m.span() for m in re.finditer(r'\S+',text)]
    class Encoding:pass
    e=Encoding();e.ids=list(range(len(spans)+(2 if add_special_tokens else 0)));e.offsets=spans;return e
  text='one two three four five six seven'
  parts=chunks(text,Tokenizer(),budget=3)
  self.assertEqual(' '.join(parts),text)
  self.assertEqual(len(parts),3)
 def test_metrics_multiple_relevant(self):
  m=metrics([['a','b','c']],[['b','c']]);self.assertEqual(m['top1'],0);self.assertEqual(m['recall_at_10'],1);self.assertEqual(m['mrr_at_10'],.5)
 def test_rrf_and_plain_negation(self):
  self.assertEqual(rrf([['a','b'],['b','a']]),['a','b'])
  text=plain({'preferences':['hiking'],'explicit_rejections':['skiing']})
  self.assertIn('explicit rejections: skiing',text)
 def test_threshold_uses_only_calibration(self):
  q=[{'id':str(i),'split':'calibration' if i<2 else 'test','relevant':['x'] if i%2==0 else []} for i in range(4)]
  a=evaluate([['x']]*4,[.8,.2,.9,.1],q)
  b=evaluate([['x']]*4,[.8,.2,.1,.9],q)
  self.assertEqual(a['metrics']['threshold'],b['metrics']['threshold'])
  self.assertNotEqual(a['metrics']['answerability_balanced_accuracy'],b['metrics']['answerability_balanced_accuracy'])
if __name__=='__main__':unittest.main()
