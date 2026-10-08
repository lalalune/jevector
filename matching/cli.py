"""Extract decision vectors, retrieve with HNSW, and enforce reciprocal conditions."""
import argparse,getpass,json
from pathlib import Path
from jevector import Client,atomic_json
from matching.engine import extract,rank_candidates,unpack

def load_gallery(query,paths):
 q=json.loads(Path(query).read_text());unpack(q);vectors={q['profile_id']:q}
 for path in paths:
  v=json.loads(Path(path).read_text());unpack(v);key=v['profile_id']
  if key in vectors:
   if vectors[key]['record_hash']!=v['record_hash']:raise ValueError('Conflicting records for profile ID: '+key)
   continue
  vectors[key]=v
 return q['profile_id'],vectors

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 e=sub.add_parser('extract');e.add_argument('--provider',required=True,choices=['clef','jev']);e.add_argument('--dimensions',type=int,choices=[64,256],default=64);e.add_argument('--account');e.add_argument('--prompt-key',action='store_true');e.add_argument('--input',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
 r=sub.add_parser('rank');r.add_argument('--query',type=Path,required=True);r.add_argument('--gallery',nargs='+',type=Path,required=True);r.add_argument('--k',type=int,default=5);r.add_argument('--method',choices=['hnsw','exact'],default='hnsw');r.add_argument('--candidates',type=int,default=32);r.add_argument('--ef',type=int,default=64);r.add_argument('--exact-fallback',action='store_true')
 args=p.parse_args()
 if args.command=='extract':
  key=getpass.getpass('Jev API key: ') if args.prompt_key else None
  result=extract(Client(args.provider,args.account,key=key,cache='runs/matching/cache'),json.loads(args.input.read_text()),args.dimensions);atomic_json(args.output,result);print('Saved',args.output)
  if result['unresolved_preferences']:print('Unrepresented optional preferences:',json.dumps(result['unresolved_preferences']))
 else:
  if args.k<1:raise ValueError('k must be positive')
  query,vectors=load_gallery(args.query,args.gallery)
  if args.method=='exact':
   rows=rank_candidates(query,vectors);out={'results':rows[:args.k],'eligible_count':len(rows),'exhaustive':True}
  else:
   from matching.ann import ProfileIndex
   out=ProfileIndex(vectors,args.ef).search(query,args.candidates,args.k,args.exact_fallback)
  out['query_id']=query;out['unresolved_preferences']=vectors[query]['unresolved_preferences']
  print(json.dumps(out,indent=2))
if __name__=='__main__':main()
