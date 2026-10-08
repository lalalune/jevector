"""Run live reciprocal extraction and reference methods against frozen synthetic pairs."""
import argparse,concurrent.futures,getpass,json,time
from pathlib import Path
import numpy as np
from jevector import Client,atomic_json,digest
from matching.schema import SCHEMA_HASH,attributes
from matching.legacy_engine import extract,unpack,rank_candidates,rank_pool,reciprocal
from evaluation.common import embed,rank,rrf,PREFIX,plain
from evaluation.lexical import BM25

def evaluate(data,rankings):
 people={p['id']:p for p in data['people']};paired=[q for q in data['query_ids'] if people[q]['kind']=='paired'];no_match=[q for q in data['query_ids'] if people[q]['kind']=='no_match'];rows=[]
 for qid in data['query_ids']:
  order=rankings[qid];q=people[qid];partner=q.get('partner');reference=data['oracle'][qid]
  assert qid not in order and len(order)==len(set(order)) and all(i in people for i in order)
  chosen=order[0] if order else None
  truth=reciprocal(q['truth'],people[chosen]['truth']) if chosen else None
  rows.append({'query_id':qid,'partner_id':partner,'top1':chosen,'ranking':order[:5],'partner_rank':order.index(partner)+1 if partner in order else None,'oracle_best':chosen in reference['best'] if chosen else False,'hard_violation':bool(truth and not truth['eligible']),'returned':bool(order),'no_match_expected':q['kind']=='no_match','truth_failure':truth if truth and not truth['eligible'] else None})
 matched={r['query_id']:r['top1']==r['partner_id'] for r in rows if r['partner_id']}
 selected=[r for r in rows if r['query_id'] in paired]
 return {'metrics':{'paired_queries':len(paired),'partner_top1':sum(matched.values())/len(paired),'partner_recall5':sum(r['partner_rank'] is not None and r['partner_rank']<=5 for r in selected)/len(paired),'oracle_best_top1':sum(r['oracle_best'] for r in selected)/len(paired),'hard_violation_count':sum(r['hard_violation'] for r in rows),'returned_count':sum(r['returned'] for r in rows),'paired_abstentions':sum(not r['returned'] for r in selected),'no_match_rejections':sum(not r['returned'] for r in rows if r['no_match_expected']),'no_match_queries':len(no_match),'both_direction_pairs':sum(matched[a] and matched[b] for a,b in data['pairs']),'pairs':len(data['pairs'])},'rows':rows}

def extraction_metrics(data,vectors,size):
 correct={k:[0,0] for k in ('self','prefer','require','exclude')}
 for person in data['people']:
  output=unpack(vectors[person['id']])
  for channel in correct:
   for key in attributes(size):
    correct[channel][1]+=1;correct[channel][0]+=output[channel][key]==person['truth'][channel][key]
 return {channel:{'correct':c,'total':n,'accuracy':c/n} for channel,(c,n) in correct.items()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--account',required=True);p.add_argument('--providers',nargs='+',default=['clef','jev']);p.add_argument('--dimensions',nargs='+',type=int,default=[64,256]);p.add_argument('--prompt-key',action='store_true');p.add_argument('--baselines-only',action='store_true');p.add_argument('--workers',type=int,default=6);args=p.parse_args();key=getpass.getpass('Jev API key: ') if args.prompt_key else None
 data=json.loads(Path('matching/data.json').read_text());protocol=json.loads(Path('matching/protocol.json').read_text());assert protocol['data_hash']==digest(data) and protocol['schema_hash']==SCHEMA_HASH
 dest=Path('runs/matching/results.json');result={'data_hash':digest(data),'schema_hash':SCHEMA_HASH,'methods':{}}
 if dest.exists():
  old=json.loads(dest.read_text())
  if old['data_hash']==digest(data) and old['schema_hash']==SCHEMA_HASH:result=old
 people=data['people'];ids=[p['id'] for p in people];lookup={key:i for i,key in enumerate(ids)}
 def save(name,orders,extra=None):
  result['methods'][name]={**evaluate(data,orders),**(extra or {})};atomic_json(dest,result);print(name,result['methods'][name]['metrics'],flush=True)
 from matching.template_control import rankings as template_rankings
 save('Template parser control',template_rankings({p['id']:p['state'] for p in people},data['query_ids']))
 own=[p['state']['about_me'] for p in people];wants=[plain(p['state']['looking_for']) for p in people]
 lexical=BM25(own);scores=np.asarray([lexical.score(text) for text in wants]);scores/=np.maximum(scores.max(axis=1,keepdims=True),1e-9);scores=(scores+scores.T)/2;np.fill_diagonal(scores,-np.inf)
 bm25={qid:rank(scores[lookup[qid]],ids)[:-1] for qid in data['query_ids']};save('BM25 reciprocal',bm25)
 a,da=embed(args.account,own);bge_scores=None
 for prefix,label in [('', 'BGE reciprocal, no prefix'),(PREFIX,'BGE reciprocal, query prefix')]:
  b,db=embed(args.account,wants,prefix=prefix);s=b@a.T;s=(s+s.T)/2;np.fill_diagonal(s,-np.inf)
  orders={qid:rank(s[lookup[qid]],ids)[:-1] for qid in data['query_ids']};save(label,orders,{'document_embedding':da,'preference_embedding':db})
  if not prefix:
   bge_scores=s.copy();fused={qid:rrf([bm25[qid],orders[qid]]) for qid in data['query_ids']};save('BM25 + BGE RRF',fused)
   reranked={qid:sorted(bm25[qid][:10],key=lambda pid:(-float(s[lookup[qid],lookup[pid]]),pid)) for qid in data['query_ids']}
   save('BM25 top-10 + BGE rerank',reranked)
 if args.baselines_only:return
 for provider in args.providers:
  client=Client(provider,args.account,key=key,cache='runs/matching/cache')
  for size in args.dimensions:
   start=time.perf_counter();folder=Path(f'runs/matching/{provider}-{size}');vectors={}
   def work(person):
    v=extract(client,person['state'],size);v['input_hash']=digest(person['state']);atomic_json(folder/(person['id']+'.json'),v);return person['id'],v
   with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    for i,(pid,v) in enumerate(pool.map(work,people)):
     vectors[pid]=v
     if (i+1)%25==0:print(provider,size,i+1,'/',len(people),flush=True)
   orders={qid:[r['id'] for r in rows] for qid,rows in rank_pool(vectors,data['query_ids']).items()}
   name=f'{provider.capitalize()} {size}'
   extra={'workers':args.workers,'extraction':extraction_metrics(data,vectors,size),'input_tokens':sum(v['input_tokens'] for v in vectors.values()),'recorded_profile_seconds_p50':float(np.median([v['request_seconds'] for v in vectors.values()])),'wall_seconds_this_run':time.perf_counter()-start}
   save(name,orders,extra)
   # Same extracted hard filters; embedding scores replace only optional preference ranking.
   filtered={qid:sorted(orders[qid],key=lambda pid:(-float(bge_scores[lookup[qid],lookup[pid]]),pid)) for qid in data['query_ids']}
   save(name+' constraints + BGE',filtered)
 print('Saved',dest)
if __name__=='__main__':main()
