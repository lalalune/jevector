import tempfile,unittest
from pathlib import Path
from search_demo.schema import dimensions,questions,VERSION
from search_demo.engine import SearchIndex

def vector(values,role='document'):
 return {'values':values,'role':role,'provider':'test','models':['test'],'schema_version':VERSION,'dimension_ids':[d['id'] for d in dimensions(len(values))]}
class Tests(unittest.TestCase):
 def test_rerank_keeps_candidate_boundary(self):
  from search_demo.rerank_baseline import rerank
  scores={'outside':1.,'a':.2,'b':.8,'c':.8}
  self.assertEqual(rerank(['a','b','c'],scores,2),['b','c'])
  self.assertEqual(rerank(['a'],scores),['a'])

 def test_schema_alignment_and_asymmetry(self):
  for n in (64,256):
   q,d=questions(n,'query'),questions(n,'document');self.assertEqual(list(q),list(d));self.assertEqual(len(q),n)
   self.assertNotEqual(next(iter(q.values()))['instructions'],next(iter(d.values()))['instructions'])
 def test_saved_hnsw_matches_exact_and_explains_contributions(self):
  with tempfile.TemporaryDirectory() as tmp:
   docs=[{'id':'a','title':'A'},{'id':'b','title':'B'}];a=[1.]+[0.]*63;b=[0.,1.]+[0.]*62
   index=SearchIndex(tmp);index.build(docs,[vector(a),vector(b)]);index=SearchIndex(tmp).load();q=vector(a,'query');r=index.search(q,2)
   self.assertEqual(r[0]['document_id'],'a');self.assertEqual([x['document_id'] for x in r],index.exact(q,2));self.assertAlmostEqual(r[0]['cosine_similarity'],1)
   self.assertAlmostEqual(sum(x['cosine_contribution'] for x in r[0]['dimensions']),1)
   with self.assertRaises(ValueError):index.search(vector(a),1)
   with self.assertRaises(ValueError):index.search({**q,'models':['wrong']},1)
 def test_zero_vectors_and_corrupted_index_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   index=SearchIndex(tmp)
   with self.assertRaises(ValueError):index.build([{'id':'a','title':'A'}],[vector([0.]*64)])
   index.build([{'id':'a','title':'A'}],[vector([1.]+[0.]*63)]);Path(tmp,'hnsw.bin').write_bytes(b'bad')
   with self.assertRaises(ValueError):SearchIndex(tmp).load()

class RequirementTests(unittest.TestCase):
 def test_coverage_extra_content_and_explicit_exclusions(self):
  with tempfile.TemporaryDirectory() as tmp:
   q=[1.]+[0.]*63
   partial=[.6]+[0.]*63
   full=[1.,1.]+[0.]*62
   index=SearchIndex(tmp);index.build([{'id':'partial','title':'Partial'},{'id':'full','title':'Full'}],[vector(partial),vector(full)])
   result=index.search(vector(q,'query'),2,scoring='coverage')
   self.assertEqual(result[0]['document_id'],'full')
   self.assertAlmostEqual(sum(d['score_contribution'] for d in result[0]['dimensions']),result[0]['score'])
   dim=index.meta['dimension_ids'][1]
   self.assertEqual(index.search(vector(q,'query'),2,excluded={dim:.5})[0]['document_id'],'partial')
   self.assertEqual(index.search(vector(q,'query'),2,scoring='coverage',excluded={dim:.5},min_score=.8),[])
   with self.assertRaises(ValueError):index.search(vector(q,'query'),required={'typo':.5})
 def test_search_effort_does_not_grow_with_corpus(self):
  with tempfile.TemporaryDirectory() as tmp:
   docs=[{'id':str(i),'title':str(i)} for i in range(100)]
   index=SearchIndex(tmp);index.build(docs,[vector([1.]+[0.]*63)]*100,ef_search=32)
   self.assertEqual(index.meta['ef_search'],32)
   self.assertEqual(SearchIndex(tmp).load().meta['ef_search'],32)

if __name__=='__main__':unittest.main()
