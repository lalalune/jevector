"""Create a fixed SciFact subset with independent public relevance labels."""
import csv,io,json,random,urllib.request,zipfile,hashlib,datetime
from pathlib import Path
from evaluation.science_schema import RUBRIC,SCHEMA_HASH
from jevector import atomic_json
URL='https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
def main():
 root=Path('runs/evaluation');root.mkdir(parents=True,exist_ok=True)
 # Persist schema and protocol before downloading or reading labels.
 protocol={'schema_hash':SCHEMA_HASH,'rubric':RUBRIC,'seed':20261006,'documents':500,'queries':80,'calibration_answered':15,'calibration_unanswerable':5,'test_answered':45,'test_unanswerable':15,'selection':'Seeded query sample; keep relevant documents for answered queries; exclude all known relevant documents for unanswerable queries; fill with seeded random distractors.','unanswerable_definition':'No labeled relevant document remains; incomplete labels can miss relevant material.','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 lock=root/'public-protocol.json'
 if not lock.exists():atomic_json(lock,protocol)
 else:
  saved=json.loads(lock.read_text())
  if {k:v for k,v in saved.items() if k!='created_at'}!={k:v for k,v in protocol.items() if k!='created_at'}:raise ValueError('Frozen protocol changed; use a new experiment version')
 archive=root/'scifact.zip'
 if not archive.exists():archive.write_bytes(urllib.request.urlopen(URL,timeout=120).read())
 with zipfile.ZipFile(archive) as z:
  docs=[json.loads(x) for x in z.read('scifact/corpus.jsonl').splitlines()]
  queries={d['_id']:d for d in map(json.loads,z.read('scifact/queries.jsonl').splitlines())}
  labels={}
  for row in csv.DictReader(io.StringIO(z.read('scifact/qrels/test.tsv').decode()),delimiter='\t'):
   if int(row['score'])>0:labels.setdefault(row['query-id'],[]).append(row['corpus-id'])
 rng=random.Random(protocol['seed']);qids=rng.sample(sorted(labels),80)
 absent=qids[60:];excluded={d for q in absent for d in labels[q]}
 # Avoid accidental removal of a positive from an answered query.
 if any(excluded.intersection(labels[q]) for q in qids[:60]):
  # Deterministic selection with disjoint known positives.
  pool=list(sorted(labels));rng.shuffle(pool);qids=[];used=set()
  for q in pool:
   if not used.intersection(labels[q]):qids.append(q);used.update(labels[q])
   if len(qids)==80:break
  absent=qids[60:];excluded={d for q in absent for d in labels[q]}
 required={d for q in qids[:60] for d in labels[q]}
 distractors=[d['_id'] for d in docs if d['_id'] not in required|excluded]
 chosen=required|set(rng.sample(distractors,500-len(required)))
 selected=[{'id':d['_id'],'title':d['title'],'body':d['text']} for d in docs if d['_id'] in chosen]
 rows=[]
 for i,qid in enumerate(qids):
  answered=i<60;cal=i<15 or 60<=i<65
  rows.append({'id':qid,'text':queries[qid]['text'],'relevant':labels[qid] if answered else [],'withheld_relevant':[] if answered else labels[qid],'split':'calibration' if cal else 'test'})
 atomic_json(root/'public-data.json',{'source':URL,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'schema_hash':SCHEMA_HASH,'documents':selected,'queries':rows,'full_corpus_count':len(docs),'license':'CC-BY-SA-4.0 (BeIR/scifact dataset card)'})
 print('Prepared',len(selected),'documents and',len(rows),'queries; schema',SCHEMA_HASH)
if __name__=='__main__':main()
