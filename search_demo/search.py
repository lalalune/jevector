"""Encode a new query and search a saved index. Prints ordered, explained hits."""
import argparse,getpass,json
from pathlib import Path
from jevector import Client,atomic_json
from search_demo.engine import encode,SearchIndex
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--query',required=True);p.add_argument('--account');p.add_argument('--prompt-key',action='store_true');p.add_argument('--k',type=int,default=5);p.add_argument('--save');p.add_argument('--scoring',choices=['cosine','coverage'],default='cosine');p.add_argument('--constraints',type=Path);p.add_argument('--min-score',type=float);args=p.parse_args();index=SearchIndex(args.index).load();key=getpass.getpass('Jev API key (hidden): ') if args.prompt_key else None
 client=Client(index.meta['provider'],args.account,model=index.meta['models'][0],key=key,cache='search_demo/cache');query=encode(client,args.query,len(index.meta['dimension_ids']),'query')
 # An unsupported query may still produce a nearest neighbor. This is an explicit
 # heuristic signal, not a calibrated relevance guarantee or automatic rejection.
 constraints=json.loads(args.constraints.read_text()) if args.constraints else {}
 if set(constraints)-{'required','excluded'}:raise ValueError('Constraints support required and excluded maps only')
 result={'query':args.query,'max_dimension_value':max(query['values']),'low_schema_activation':max(query['values'])<.5,'results':index.search(query,args.k,scoring=args.scoring,min_score=args.min_score,**constraints),'query_vector':query}
 if args.save:atomic_json(args.save,result)
 print(json.dumps({k:v for k,v in result.items() if k!='query_vector'},indent=2))
