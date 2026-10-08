"""Run paired retrieval and extraction checks with either provider and schema."""
import argparse, concurrent.futures, getpass, json, math, statistics, time
from pathlib import Path
from jevector import Client, atomic_json, digest, distance
from schema import schema, HOBBIES

def expected_for(expected,size):
 if size==256:return expected
 ids={d['id'] for d in schema(size)};result={k:v for k,v in expected.items() if k in ids}
 for group,hs in HOBBIES.items():
  labels=[expected.get('interest.'+h) for h in hs]
  if 'yes' in labels:result['interest_group.'+group]='yes'
  elif all(x=='no' for x in labels):result['interest_group.'+group]='no'
 return result

def retrieval(records,people,size,groups=None,metric=distance):
 dims=schema(size);rows=[]
 for p in people:
  query=records[p['id']+'_b'];scored=[]
  for g in people:
   dist=metric(query,records[g['id']+'_a'],dims,groups)
   if dist is not None:scored.append((dist,g['id']))
  scored.sort();own=next((d for d,k in scored if k==p['id']),None)
  if own is None:rank=None;ties=0
  else:
   rank=1+sum(d<own-1e-9 for d,k in scored);ties=sum(abs(d-own)<1e-9 for d,k in scored)
  rows.append({'query':p['id'],'top_match':scored[0][1] if scored else None,'rank':rank,'ties_at_own_distance':ties,'correct_unique_top1':rank==1 and ties==1,'own_distance':own,'nearest_other':min((d for d,k in scored if k!=p['id']),default=None)})
 return {'unique_top1':sum(r['correct_unique_top1'] for r in rows)/len(rows),'top3':sum(r['rank'] is not None and r['rank']+r['ties_at_own_distance']-1<=3 for r in rows)/len(rows),'mrr':sum(1/(r['rank']+(r['ties_at_own_distance']-1)/2) if r['rank'] else 0 for r in rows)/len(rows),'abstentions':sum(r['rank'] is None for r in rows),'rows':rows}

def summarize(records,dataset,size):
 people=dataset['people'];cases=[]
 for p in people:
  for view in ('a','b'):
   answer=records[p['id']+'_'+view]['answers']
   for key,label in expected_for(p.get('expected_by_view',{}).get(view,p['expected']),size).items():cases.append({'key':key,'expected':label,'predicted':answer[key]['choice'],'p_expected':answer[key]['probabilities'][label]})
 groups={}
 for group,prefix in [('interests','interest'),('relationships','relationship.'),('lifestyle','lifestyle.')]:
  subset=[c for c in cases if c['key'].startswith(prefix)];groups[group]={'count':len(subset),'accuracy':sum(c['expected']==c['predicted'] for c in subset)/len(subset)}
 controls={}
 for c in dataset['controls']:
  rec=records['control_'+c['id']];labels=expected_for(c['expected'],size)
  checks={k:{'expected':v,'predicted':rec['answers'][k]['choice'],'correct':v==rec['answers'][k]['choice']} for k,v in labels.items()}
  controls[c['id']]={'observed_dimensions':sum(rec['mask']),'unknown_fraction':sum(a['choice']=='unknown' for a in rec['answers'].values())/size,'checks':checks}
 return {'people':len(people),'dimensions':size,'retrieval':retrieval(records,people,size),'personality_only_retrieval':retrieval(records,people,size,['personality']),'explicit_preference_accuracy':sum(c['expected']==c['predicted'] for c in cases)/len(cases),'extraction_groups':groups,'extraction_errors':[c for c in cases if c['expected']!=c['predicted']],'controls':controls,'models':sorted({m for r in records.values() for m in r['models']}),'total_input_tokens':sum(r['input_tokens'] for r in records.values()),'profile_latency_median_seconds':statistics.median(r['elapsed_seconds'] for k,r in records.items() if not k.startswith('control_')),'profile_latency_mean_seconds':statistics.mean(r['elapsed_seconds'] for k,r in records.items() if not k.startswith('control_'))}

def run(provider,size,args,key=None):
 dataset=json.loads(Path(args.dataset).read_text())
 if args.limit:dataset['people']=dataset['people'][:args.limit]
 folder=Path(args.output)/f'{provider}-{size}'
 if args.batch_size:folder=folder.with_name(folder.name+f'-batch{args.batch_size}')
 client=Client(provider,args.account,key=key,batch_size=args.batch_size)
 jobs=[(p['id']+'_'+view,state) for p in dataset['people'] for view,state in p['views'].items()]+[('control_'+c['id'],c['state']) for c in dataset['controls']]
 records={};started=time.monotonic()
 def extract(job):
  name,state=job;record=client.extract(state,size);atomic_json(folder/'profiles'/(name+'.json'),record);return name,record
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
  for f in concurrent.futures.as_completed([pool.submit(extract,j) for j in jobs]):
   name,record=f.result();records[name]=record
   print(f'{provider}-{size} {len(records)}/{len(jobs)} {name} {sum(record["mask"])}/{size} observed',flush=True)
 summary=summarize(records,dataset,size);summary.update({'provider':provider,'batch_size':client.batch_size,'dataset_hash':digest(dataset),'wall_seconds':time.monotonic()-started,'estimated_cost_usd':summary['total_input_tokens']*(.09 if provider=='clef' else .042)/1e6})
 atomic_json(folder/'metrics.json',summary)
 print(json.dumps({k:v for k,v in summary.items() if k not in ('retrieval','personality_only_retrieval','extraction_errors','controls')},indent=2),flush=True)
 return summary

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--provider',choices=['clef','jev','both'],required=True);p.add_argument('--dimensions',choices=['64','256','both'],default='both');p.add_argument('--account');p.add_argument('--prompt-key',action='store_true');p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--dataset',default='data/synthetic.json');p.add_argument('--output',default='runs');args=p.parse_args()
 key=getpass.getpass('Jev API key (hidden): ') if args.prompt_key else None
 for provider in (['clef','jev'] if args.provider=='both' else [args.provider]):
  for size in ([64,256] if args.dimensions=='both' else [int(args.dimensions)]):run(provider,size,args,key)
if __name__=='__main__':main()
