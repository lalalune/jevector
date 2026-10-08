"""Run the fixed v2.1 extraction and coverage matcher on original and fresh fixtures."""
import argparse,getpass
from benchmark import run
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--provider',choices=['clef','jev','both'],required=True);p.add_argument('--account');p.add_argument('--prompt-key',action='store_true');p.add_argument('--workers',type=int,default=4);args=p.parse_args()
 key=getpass.getpass('Jev API key (hidden): ') if args.prompt_key else None
 for provider in (['clef','jev'] if args.provider=='both' else [args.provider]):
  for dataset,output in [('data/synthetic.json','runs/v21'),('data/fresh-stress.json','runs/fresh-v21')]:
   config=argparse.Namespace(dataset=dataset,output=output,limit=None,batch_size=None,workers=args.workers,account=args.account)
   for size in (64,256):run(provider,size,config,key)
