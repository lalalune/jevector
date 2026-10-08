"""Recompute reciprocal metrics, extraction errors, and pair-level failures."""
import json
from pathlib import Path
from matching.legacy_engine import unpack,rank_candidates,rank_pool,reciprocal
from matching.run import evaluate
from matching.schema import attributes,SCHEMA_HASH
from jevector import digest,atomic_json

def main():
 data=json.loads(Path('matching/data.json').read_text());result=json.loads(Path('runs/matching/results.json').read_text());assert result['data_hash']==digest(data) and result['schema_hash']==SCHEMA_HASH
 outputs={}
 for provider in ('clef','jev'):
  for size in (64,256):
   name=f'{provider.capitalize()} {size}';folder=Path(f'runs/matching/{provider}-{size}')
   vectors={p['id']:json.loads((folder/(p['id']+'.json')).read_text()) for p in data['people']}
   for p in data['people']:assert vectors[p['id']]['input_hash']==digest(p['state'])
   orders={qid:[row['id'] for row in rows] for qid,rows in rank_pool(vectors,data['query_ids']).items()}
   measured=evaluate(data,orders);assert measured['metrics']==result['methods'][name]['metrics']
   errors=[];channels={}
   for channel in ('self','prefer','require','exclude'):
    tp=fp=fn=known_count=known_correct=unknown_count=unknown_correct=0
    for person in data['people']:
     decoded=unpack(vectors[person['id']])[channel]
     for key in attributes(size):
      expected=person['truth'][channel][key];predicted=decoded[key]
      tp+=expected is True and predicted is True;fp+=expected is not True and predicted is True;fn+=expected is True and predicted is not True
      if expected is None:unknown_count+=1;unknown_correct+=predicted is None
      else:known_count+=1;known_correct+=predicted==expected
      if expected!=predicted:errors.append({'profile_id':person['id'],'channel':channel,'attribute':key,'expected':expected,'predicted':predicted})
    precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
    channels[channel]={'positive_precision':precision,'positive_recall':recall,'positive_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,'known_accuracy':known_correct/known_count if known_count else None,'unknown_accuracy':unknown_correct/unknown_count if unknown_count else None,'known_count':known_count,'unknown_count':unknown_count}
   outputs[name]={'extraction':channels,'errors':errors,'failures':[row for row in measured['rows'] if row['hard_violation'] or row['partner_id'] and row['top1']!=row['partner_id']]}
 for name,method in result['methods'].items():
  # Rows preserve only top five. Verify top-1 safety and pair counts separately from full-rank metrics.
  assert len(method['rows'])==36
  assert method['metrics']['hard_violation_count']==sum(row['hard_violation'] for row in method['rows'])
  assert method['metrics']['returned_count']==sum(row['returned'] for row in method['rows'])
 atomic_json('runs/matching/analysis.json',{'data_hash':digest(data),'schema_hash':SCHEMA_HASH,'methods':outputs})
 print('Verified four vector methods, 656 profile input hashes, all top-result checks, and extraction errors.')
if __name__=='__main__':main()
