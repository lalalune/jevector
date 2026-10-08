import unittest
from matching.comparison import metrics

class ComparisonTests(unittest.TestCase):
 def test_rank_cutoff_missing_and_ties(self):
  data={'people':[{'id':'q','kind':'paired','split':'validation','partner':'p'},{'id':'n','kind':'no_match','split':'validation'}],'query_ids':['q','n']}
  rows=[{'id':'a','score':1.},{'id':'p','score':1.},{'id':'b','score':0.}]
  v=metrics(data,{'q':rows},'validation')
  self.assertEqual(v['queries'],1);self.assertEqual(v['top1'],0);self.assertEqual(v['top5'],1);self.assertEqual(v['mrr10'],.5);self.assertEqual(v['tie_top1'],.5)
  rows=[{'id':str(i),'score':20-i} for i in range(10)]+[{'id':'p','score':0.}]
  v=metrics(data,{'q':rows},'validation')
  self.assertEqual(v['top10'],0);self.assertEqual(v['mrr10'],0);self.assertEqual(v['ndcg10'],0)
  self.assertEqual(metrics(data,{'q':[]},'validation')['tie_top1'],0)

if __name__=='__main__':unittest.main()
