"""Compare old/new matchers on identical vectors and report fresh live evaluations."""
import json, hashlib
from pathlib import Path
from benchmark import retrieval
from jevector import distance, legacy_distance, atomic_json, MATCHER_VERSION, digest
from schema import VERSION
NAMES=['clef-64','clef-256','jev-64','jev-256']
def read(path):return json.loads(Path(path).read_text())
def profiles(folder):return {p.stem:read(p) for p in Path(folder,'profiles').glob('*.json')}
def controls(m):
 checks=[c for group in m['controls'].values() for c in group['checks'].values()]
 return f"{sum(c['correct'] for c in checks)}/{len(checks)}"
def count(x,n):return round(x*n)
def main():
 original=read('data/synthetic.json');fresh=read('data/fresh-stress.json');results={}
 for name in NAMES:
  size=int(name.split('-')[1]);old=read(f'runs/{name}/metrics.json');new=read(f'runs/v21/{name}/metrics.json');fm=read(f'runs/fresh-v21/{name}/metrics.json')
  rescored=retrieval(profiles(f'runs/{name}'),original['people'],size)
  old_fresh=retrieval(profiles(f'runs/fresh-v21/{name}'),fresh['people'],size,metric=legacy_distance)
  results[name]={'old_top1':old['retrieval']['unique_top1'],'old_answers_new_matcher':rescored['unique_top1'],'new_live_top1':new['retrieval']['unique_top1'],'fresh_old_matcher':old_fresh['unique_top1'],'fresh_new_matcher':fm['retrieval']['unique_top1'],'old_extraction_accuracy':old['explicit_preference_accuracy'],'new_extraction_accuracy':new['explicit_preference_accuracy'],'fresh_extraction_accuracy':fm['explicit_preference_accuracy'],'new_controls':controls(new),'fresh_controls':controls(fm),'fresh_legacy_rows':old_fresh['rows']}
 atomic_json('runs/improvements.json',{'schema_version':VERSION,'matcher_version':MATCHER_VERSION,'original_fixture_hash':digest(original),'fresh_fixture_hash':digest(fresh),'results':results,'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['jevector.py','schema.py','benchmark.py','fresh_data.py']}})
 from documentation import render_reports
 render_reports()
 print('Updated current profile report.')
if __name__=='__main__':main()
