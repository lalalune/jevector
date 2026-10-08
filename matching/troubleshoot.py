"""Offline decoder ablations. Preserve the benchmark; do not select a deployment threshold."""
import errno,json,time
from pathlib import Path
from jevector import atomic_json,digest
from matching.legacy_engine import unpack,reciprocal
from matching.schema import SCHEMA_HASH,attributes
from matching.run import evaluate

def read_json(path):
 for attempt in range(4):
  try:return json.loads(Path(path).read_text())
  except OSError as error:
   if error.errno not in (errno.ENFILE,errno.EMFILE) or attempt==3:raise
   time.sleep(.5*(attempt+1))

MODES=('original','threshold_0.60','threshold_0.55','condition_choice','selected_choice')

def decode(vector,mode):
 result=unpack(vector)
 if mode=='original':return result
 for i,key in enumerate(vector['dimension_ids']):
  channel,attribute=key.split('.',1);answer=vector['answers'][key]
  if mode=='selected_choice' or mode=='condition_choice' and channel!='self':
   result[channel][attribute]={'yes':True,'no':False}.get(answer['choice'])
  elif mode.startswith('threshold_'):
   cutoff=float(mode.split('_')[1]);value=vector['values'][i]
   result[channel][attribute]=(True if value>=cutoff else False if value<=1-cutoff else None) if vector['known'][i] else None
 return result

def main():
 data=json.loads(Path('matching/data.json').read_text());people={p['id']:p for p in data['people']};reference=json.loads(Path('runs/matching/results.json').read_text())
 assert reference['schema_hash']==SCHEMA_HASH and reference['data_hash']==digest(data)
 # Ground truth is used only after ranking, to score every directed query/candidate pair.
 truth={(q,p):reciprocal(people[q]['truth'],people[p]['truth'])['eligible'] for q in data['query_ids'] for p in people if q!=p}
 output={'schema_hash':SCHEMA_HASH,'data_hash':digest(data),'scope':'Post-hoc development-data diagnosis; no threshold selected or independent validation claimed.','methods':{}}
 for provider in ('clef','jev'):
  for size in (64,256):
   name=f'{provider.capitalize()} {size}';vectors={pid:read_json(Path(f'runs/matching/{provider}-{size}')/(pid+'.json')) for pid in people}
   assert all(vectors[pid]['input_hash']==digest(people[pid]['state']) for pid in people)
   entry={'ablations':{},'blocked_partner_conditions':[],'extraction':{}}
   for mode in MODES:
    decoded={pid:decode(v,mode) for pid,v in vectors.items()};rankings={};tp=fp=fn=tn=0
    for q in data['query_ids']:
     rows=[]
     for pid in people:
      if pid==q:continue
      m=reciprocal(decoded[q],decoded[pid]);expected=truth[q,pid]
      tp+=m['eligible'] and expected;fp+=m['eligible'] and not expected;fn+=not m['eligible'] and expected;tn+=not m['eligible'] and not expected
      if m['eligible']:rows.append((pid,m['score']))
     rankings[q]=[pid for pid,score in sorted(rows,key=lambda row:(-row[1],row[0]))]
    metrics=evaluate(data,rankings)['metrics']
    if mode=='original':assert metrics==reference['methods'][name]['metrics']
    entry['ablations'][mode]={'metrics':metrics,'all_candidate_eligibility':{'true_accepts':tp,'false_accepts':fp,'false_rejects':fn,'true_rejects':tn}}
   decoded={pid:unpack(v) for pid,v in vectors.items()}
   for channel in ('self','prefer','require','exclude'):
    total=choice_errors=decoded_errors=threshold_abstentions=0
    for pid,v in vectors.items():
     for key in attributes(size):
      a=v['answers'][channel+'.'+key];expected=people[pid]['truth'][channel][key];selected={'yes':True,'no':False}.get(a['choice']);actual=decoded[pid][channel][key]
      total+=1;choice_errors+=selected!=expected;decoded_errors+=actual!=expected;threshold_abstentions+=selected is not None and actual is None
    entry['extraction'][channel]={'total':total,'selected_answer_errors':choice_errors,'decoded_errors':decoded_errors,'threshold_created_abstentions':threshold_abstentions}
   for a,b in data['pairs']:
    for q,candidate in ((a,b),(b,a)):
     for key in attributes(size):
      req=decoded[q]['require'][key];exc=decoded[q]['exclude'][key];actual=decoded[candidate]['self'][key]
      keys=[]
      if req is None:keys.append((q,'require.'+key,'uncertain condition'))
      if exc is None:keys.append((q,'exclude.'+key,'uncertain condition'))
      if (req is True or exc is True) and actual is None:keys.append((candidate,'self.'+key,'unknown candidate evidence'))
      for pid,dimension,reason in keys:
       answer=vectors[pid]['answers'][dimension];ch,k=dimension.split('.',1)
       entry['blocked_partner_conditions'].append({'requester':q,'candidate':candidate,'profile':pid,'dimension':dimension,'reason':reason,'source_label':people[pid]['truth'][ch][k],'answer':answer})
   output['methods'][name]=entry
   print(name,{mode:r['metrics']['partner_top1'] for mode,r in entry['ablations'].items()},flush=True)
 atomic_json('runs/matching/troubleshooting.json',output)
 lines=['# Clef decoder diagnosis','','These are post-hoc tests on the existing synthetic development fixture. The original benchmark is unchanged.','', '| Model | Decoder | Partner top-1 | False accepts / 5,804 incompatible comparisons | False rejects / 64 eligible comparisons |','|---|---|---:|---:|---:|']
 for name,entry in output['methods'].items():
  for mode,r in entry['ablations'].items():
   m=r['metrics'];c=r['all_candidate_eligibility'];lines.append(f"| {name} | {mode} | {round(m['partner_top1']*32)}/32 | {c['false_accepts']} | {c['false_rejects']} |")
 lines+=['','## Cause','','The original decoder requires a numeric value of at least 0.7 for yes or at most 0.3 for no.','Other values become unknown. Any unknown required or excluded condition rejects the pair.','This also rejects pairs when a low-confidence no concerns an unstated condition.','The larger schema has more opportunities for this rejection.','','## Remaining extraction error','','Clef accepts one incompatible lower-ranked candidate under both original and selected-answer decoding.','That profile omits its alcohol-free preference. Clef infers yes instead of unknown.','The intended partner remains first, so top-1 violation counts do not expose this error.','The all-candidate table includes it. Counts refer to directed query/candidate comparisons.','','## Interpretation','','`condition_choice` uses selected answers for partner conditions and retains the original self-evidence checks.','`selected_choice` uses selected answers for all channels. Explicit mixed and unknown answers remain unknown.','The threshold variants retain the original evidence mask and change only the yes/no cutoff.','No API requests, prompt changes, or ground-truth filtering are used. Labels score the results only.','','Equal cutoffs do not establish equal error rates across providers. Model probabilities have not been calibrated here.','Clef still has more selected-answer extraction errors. Improved match scores do not remove those errors.','Do not deploy a lower threshold based on these pairs. Validate it on independently generated profiles and missing-evidence cases first.','','Run `search_demo/python -m matching.troubleshoot` to reproduce these tables.','Detailed answers and channel error counts are in `runs/matching/troubleshooting.json`.']
 Path('matching/TROUBLESHOOTING.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
