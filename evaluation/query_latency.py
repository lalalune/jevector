"""Record sequential single-query BGE requests separately from batched indexing."""
import argparse,json,time
from pathlib import Path
import numpy as np
from evaluation.common import cached_call,BGE,PREFIX
from jevector import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--account',required=True);a=p.parse_args();data=json.loads(Path('runs/evaluation/public-data.json').read_text());rows=[]
 for q in [q for q in data['queries'] if q['split']=='test'][:10]:
  r,hit=cached_call(a.account,BGE,{'text':[PREFIX+q['text']],'pooling':'cls'});rows.append({'query_id':q['id'],'recorded_seconds':r['elapsed_seconds'],'cache_hit_this_run':hit})
 atomic_json('runs/evaluation/query-latency.json',{'method':'BGE-small','sequential_requests':10,'p50_seconds':float(np.median([r['recorded_seconds'] for r in rows])),'p95_seconds':float(np.percentile([r['recorded_seconds'] for r in rows],95)),'rows':rows})
if __name__=='__main__':main()
