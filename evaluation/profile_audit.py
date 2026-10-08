"""Profile baseline sensitivity audit: no hidden truncation or winner selection."""
import argparse,json
from pathlib import Path
from evaluation.common import embed,plain,rank,metrics,rrf,PREFIX,tokenizer
from search_demo.run import bm25
from jevector import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--account',required=True);args=p.parse_args();result={}
 for group,file in [('original','data/synthetic.json'),('additional','data/fresh-stress.json')]:
  data=json.loads(Path(file).read_text());people=data['people'];ids=[p['id'] for p in people];rels=[[i] for i in ids];variants={};t=tokenizer()
  old=[json.dumps(p['views'][v],ensure_ascii=False) for v in ('a','b') for p in people]
  old_lengths=[len(t.encode((PREFIX if i>=len(people) else '')+text).ids) for i,text in enumerate(old)]
  for form in ('json','plain'):
   texts={v:[json.dumps(p['views'][v],ensure_ascii=False) if form=='json' else plain(p['views'][v]) for p in people] for v in ('a','b')}
   docs=[{'id':i,'title':'','body':text} for i,text in zip(ids,texts['a'])]
   lexical=[bm25(docs,q) for q in texts['b']]
   variants['BM25 '+form]={'metrics':metrics(lexical,rels),'rankings':lexical}
   a,da=embed(args.account,texts['a'])
   for prefix,label in [('', 'no prefix'),(PREFIX,'query prefix')]:
    b,db=embed(args.account,texts['b'],prefix=prefix);scores=b@a.T;orders=[rank(s,ids) for s in scores]
    name=f'BGE {form}, {label}, chunked'
    variants[name]={'metrics':metrics(orders,rels),'rankings':orders,'document_embedding':da,'query_embedding':db}
    if form=='plain' and not prefix:
     fused=[rrf([lexical[i],orders[i]]) for i in range(len(ids))]
     variants['BM25 + BGE RRF']={'metrics':metrics(fused,rels),'rankings':fused}
    print(group,name,variants[name]['metrics'],flush=True)
  result[group]={'people':len(ids),'old_inputs_over_512':sum(x>512 for x in old_lengths),'old_input_count':len(old_lengths),'old_max_tokens':max(old_lengths),'variants':variants}
  atomic_json('runs/evaluation/profile-audit.json',result)
if __name__=='__main__':main()
