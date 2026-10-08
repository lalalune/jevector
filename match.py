"""Rank saved profiles by evidence-aware similarity, with an abstention threshold."""
import argparse, json
from pathlib import Path
from jevector import distance, MATCHER_VERSION
from schema import schema, VERSION

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--query',type=Path,required=True);p.add_argument('--gallery',type=Path,nargs='+',required=True);p.add_argument('--min-shared',type=int,default=8);args=p.parse_args()
 query=json.loads(args.query.read_text());size=len(query['dimension_ids']);dims=schema(size);rows=[]
 for path in args.gallery:
  candidate=json.loads(path.read_text())
  if candidate['dimension_ids']!=query['dimension_ids'] or candidate['schema_version']!=query['schema_version']:raise ValueError('Cannot match different schemas')
  if candidate['provider']!=query['provider'] or candidate['models']!=query['models']:raise ValueError('Cross-provider/model vector distances are uncalibrated; use the same provider and model')
  score=distance(query,candidate,dims,min_shared=args.min_shared)
  rows.append({'profile':str(path),'distance':score,'shared_observed_dimensions':sum(x and y for x,y in zip(query['mask'],candidate['mask'])),'status':'ranked' if score is not None else 'insufficient_evidence'})
 rows.sort(key=lambda r:r['distance'] if r['distance'] is not None else float('inf'));print(json.dumps({'matcher_version':MATCHER_VERSION,'matches':rows},indent=2))
if __name__=='__main__':main()
