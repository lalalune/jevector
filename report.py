"""Update the saved comparison and the profile reports."""
import json
from pathlib import Path
from jevector import atomic_json
from documentation import render_reports
ORDER=['clef-64','clef-256','jev-64','jev-256']
if __name__=='__main__':
 metrics={n:json.loads(Path('runs',n,'metrics.json').read_text()) for n in ORDER}
 robust=json.loads(Path('runs/robustness.json').read_text());diagnostics={}
 for name in ORDER:
  profiles=[json.loads(p.read_text()) for p in Path('runs',name,'profiles').glob('*.json')]
  diagnostics[name]={'profiles':len(profiles),'choice_argmax_disagreements':sum(len(r['diagnostics']['choice_argmax_disagreements']) for r in profiles),'rounded_probability_sum_deviations':sum(len(r['diagnostics']['probability_sum_deviations']) for r in profiles)}
 atomic_json('runs/comparison.json',{'metrics':metrics,'robustness':robust,'diagnostics':diagnostics});render_reports()
 print('Updated profile reports.')
