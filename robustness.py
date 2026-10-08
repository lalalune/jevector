"""Small live repeat/order/batching probes, separate from main retrieval benchmark."""
import argparse, getpass, json, math
from pathlib import Path
from jevector import Client, atomic_json, normalize
from schema import schema, questions

def delta(a,b):
 shared=[i for i,(x,y) in enumerate(zip(a['mask'],b['mask'])) if x and y]
 return {'shared_observed_dimensions':len(shared),'observed_value_rms':math.sqrt(sum((a['values'][i]-b['values'][i])**2 for i in shared)/len(shared)) if shared else None,'value_rms_including_masked':math.sqrt(sum((x-y)**2 for x,y in zip(a['values'],b['values']))/len(a['values'])),'mask_flips':sum(x!=y for x,y in zip(a['mask'],b['mask'])),'choice_flips':sum(a['answers'][k]['choice']!=b['answers'][k]['choice'] for k in a['answers'])}
def probe(client,state,size):
 original=client.extract(state,size);repeat=client.extract(state,size,nonce='repeat-probe-v2')
 results={'repeat':delta(original,repeat)}
 for name,items in [('question_order',list(reversed(list(questions(size).items())))),('option_order',[(k,{**q,'criteria':dict(reversed(list(q['criteria'].items())))}) for k,q in questions(size).items()])]:
  answers={}
  for i in range(0,len(items),client.batch_size):answers.update(client.call(state,dict(items[i:i+client.batch_size]))['response']['answers'])
  other={**normalize(answers,schema(size)),'answers':answers};results[name]=delta(original,other)
 return results
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--account',required=True);p.add_argument('--prompt-key',action='store_true');args=p.parse_args()
 key=getpass.getpass('Jev API key (hidden): ') if args.prompt_key else None
 people=json.loads(Path('data/synthetic.json').read_text())['people'];state=people[0]['views']['a'];result={}
 for provider in ('clef','jev'):
  client=Client(provider,args.account,key=key)
  for size in (64,256):
   result[f'{provider}-{size}']=probe(client,state,size);print(provider,size,result[f'{provider}-{size}'],flush=True)
 single=Client('jev',key=key);batched=Client('jev',key=key,batch_size=64)
 result['jev_256_single_vs_four_batches']=[]
 for p in people[:4]:
  a=single.extract(p['views']['a'],256);b=batched.extract(p['views']['a'],256)
  result['jev_256_single_vs_four_batches'].append({'person':p['id'],**delta(a,b)})
 atomic_json('runs/robustness.json',result)
