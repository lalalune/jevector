"""Precomputed BM25 reference (k1=1.2, b=0.75), shared tokenizer."""
import collections,math,re
import numpy as np
class BM25:
 def __init__(self,texts):
  corpus=[self.tokens(t) for t in texts];self.tf=[collections.Counter(t) for t in corpus];self.df=collections.Counter(t for doc in corpus for t in set(doc));self.length=np.array([len(t) for t in corpus]);self.avg=max(float(self.length.mean()),1);self.n=len(texts)
 @staticmethod
 def tokens(text):return re.findall(r'[a-z0-9]+',text.lower())
 def score(self,text):
  scores=np.zeros(self.n)
  for term in self.tokens(text):
   df=self.df[term];idf=math.log(1+(self.n-df+.5)/(df+.5));f=np.array([tf[term] for tf in self.tf]);scores+=idf*f*2.2/(f+1.2*(.25+.75*self.length/self.avg))
  return scores
