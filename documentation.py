"""Generate profile reports from saved results."""
import html,json,re
from pathlib import Path

def read(path):return json.loads(Path(path).read_text())
def html_report(title,lines):
 out=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+html.escape(title)+'</title><style>body{font:15px system-ui;max-width:1100px;margin:36px auto;padding:0 24px;color:#172b42}table{border-collapse:collapse;display:block;overflow:auto;margin:20px 0}td,th{padding:10px;border:1px solid #dce3ea;text-align:left}p{line-height:1.5}pre{padding:16px;background:#f2f5f8;overflow:auto}a{color:#245b93}</style>']
 table=False;code=False
 for line in lines:
  if line.startswith('```'):
   out.append('</pre>' if code else '<pre>');code=not code;continue
  if code:out.append(html.escape(line)+'\n');continue
  if line.startswith('|'):
   if not table:out.append('<table>');table=True
   if re.fullmatch(r'[| :\-]+',line):continue
   out.append('<tr>'+''.join('<td>'+html.escape(cell.strip())+'</td>' for cell in line.strip('|').split('|'))+'</tr>');continue
  if table:out.append('</table>');table=False
  if not line:continue
  if line.startswith('#'):
   level=len(line)-len(line.lstrip('#'));out.append(f'<h{level}>'+html.escape(line[level:].strip())+f'</h{level}>')
  else:out.append('<p>'+html.escape(line)+'</p>')
 if table:out.append('</table>')
 out.append('<p><a href="README.md">Instructions</a> · <a href="search_demo/index.html">Document search</a></p></html>')
 return ''.join(out)
def save(md,viewer,lines):
 Path(md).write_text('\n'.join(lines)+'\n');Path(viewer).write_text(html_report(lines[0].lstrip('# '),lines))
def render_reports():
 names=['clef-64','clef-256','jev-64','jev-256'];m={n:read(f'runs/{n}/metrics.json') for n in names}
 lines=['# Profile test results: version 2.0','','Test set: 32 synthetic people with two descriptions each. Each configuration also has four controls.','','## Results','','| Configuration | Correct first matches | Preference accuracy | Median request time | Estimated cost |','|---|---:|---:|---:|---:|']
 for n,d in m.items():lines.append(f"| {n} | {round(d['retrieval']['unique_top1']*32)}/32 | {d['explicit_preference_accuracy']:.2%} | {d['profile_latency_median_seconds']:.3f} s | ${d['estimated_cost_usd']:.4f} |")
 lines+=['','## Definitions','','Top-1 is the proportion of queries with the correct profile first.','Preference accuracy compares model answers with the stated preference labels.','Request time includes network time. Detailed Clef profiles require four sequential requests.','Cost uses published input-token rates. It is not a billing statement.','','## Controls','']
 for n,d in m.items():
  checks=[x for c in d['controls'].values() for x in c['checks'].values()]
  lines.append(f"- {n}: {sum(x['correct'] for x in checks)}/{len(checks)} labeled control answers are correct.")
  for cn,c in d['controls'].items():
   for key,x in c['checks'].items():
    if not x['correct']:lines.append(f"- {n}, {cn}, {key}: expected {x['expected']}; received {x['predicted']}.")
 lines+=['','## Repeat tests','','| Configuration | Test | Shared-value RMS difference | Mask changes | Choice changes |','|---|---|---:|---:|---:|']
 robust=read('runs/robustness.json')
 for n in names:
  for test,r in robust[n].items():lines.append(f"| {n} | {test} | {r['observed_value_rms']:.4f} | {r['mask_flips']} | {r['choice_flips']} |")
 lines+=['','## Limits','','The test data is synthetic. The descriptions are related text variants.','The results do not establish personality accuracy or relationship compatibility.','Question order and request grouping can change the output.','The old matcher can favor profiles with little shared evidence.','Current results are in IMPROVEMENTS.md.','','## Data','','Saved results: runs/<configuration>/metrics.json.','Saved questions: data/archive/schema-64-v2.0.json and data/archive/schema-256-v2.0.json.','Input-token rates: Clef-flash $0.09 per million; Jev $0.042 per million.']
 save('BENCHMARK.md','comparison.html',lines)
 r=read('runs/improvements.json')['results'];lines=['# Profile test results: version 2.1','','Schema: jevector-2.1. Matcher: coverage-v1.','','## Original set','','| Configuration | Previous matches | New matcher with old answers | New live matches |','|---|---:|---:|---:|']
 for n,d in r.items():lines.append(f"| {n} | {round(d['old_top1']*32)}/32 | {round(d['old_answers_new_matcher']*32)}/32 | {round(d['new_live_top1']*32)}/32 |")
 lines+=['','## Additional set','','The additional set has 16 synthetic people. Alternate descriptions omit three interest labels.','','| Configuration | Old matcher | New matcher | Preference accuracy | Correct controls |','|---|---:|---:|---:|---:|']
 for n,d in r.items():lines.append(f"| {n} | {round(d['fresh_old_matcher']*16)}/16 | {round(d['fresh_new_matcher']*16)}/16 | {d['fresh_extraction_accuracy']:.2%} | {d['fresh_controls']} |")
 lines+=['','## Preference extraction','','| Configuration | Previous accuracy | Current accuracy |','|---|---:|---:|']
 for n,d in r.items():lines.append(f"| {n} | {d['old_extraction_accuracy']:.2%} | {d['new_extraction_accuracy']:.2%} |")
 lines+=['','## Calculation','','The matcher compares shared dimensions. It gives equal weight to each group with evidence on either side.','Each group receives the additional cost `0.25 × (1 − shared_evidence / union_evidence)`.','The result is the square root of the mean group cost.','The matcher requires at least eight shared dimensions.','Missing evidence is not a negative preference.','','## Remaining errors','','Jev returns no for the contradictory dating-intent control. The expected answer is mixed.','Preference extraction accuracy is lower on some tests.','The original set informed the change. It is development evidence.','The additional set is synthetic. It is not independent human validation.','','## Verification','','The tests checked 432 complete profile outputs.','The tests checked missing evidence, equal scores, group weights, response validation, and cache reuse.','The data does not establish production accuracy.','','## Commands','','```sh','python3 -m unittest -v test_jevector','python3 run_improvements.py --provider both --account "$CLOUDFLARE_ACCOUNT_ID" --prompt-key','python3 improvement_report.py','```','','## Files','','Original-set outputs: runs/v21/.','Additional-set outputs: runs/fresh-v21/.','Comparison data: runs/improvements.json.']
 save('IMPROVEMENTS.md','improvements.html',lines)
 first=read('results/metrics.json');lines=['# Initial personality test','','Model: Cloudflare Clef-flash. Input: 12 synthetic people with two descriptions each. Vector size: 64.','','## Results','',f"Correct first matches: {round(first['top1_accuracy']*12)}/12.",f"Top-three accuracy: {first['top3_accuracy']:.0%}.",f"TF-IDF first-match accuracy: {first['lexical_tfidf_baseline']['top1_accuracy']:.0%}.",'','| Query | First match | Correct rank | Correct distance | Closest other distance |','|---|---|---:|---:|---:|']
 for d in first['matches']:lines.append(f"| {d['person']} | {d['matched']} | {d['rank']} | {d['own_distance']:.4f} | {d['nearest_other_distance']:.4f} |")
 lines+=['','## Method','','Each dimension uses an expected score from 0 to 4. The program divides the score by four.','The test uses equal-weight root-mean-square distance.','The midpoint combines mixed evidence and missing evidence.','Later schemas separate unknown from mixed evidence.','','## Limits','','The descriptions are synthetic text variants. The test measures retrieval consistency.','It does not establish personality accuracy or interpersonal compatibility.','Raw responses and vectors are in results/.']
 save('REPORT.md','report.html',lines)
 # Retain the original viewer's complete dimension table.
 v=read('results/vectors.json');rows=['<h2>Dimension values</h2><p>Values range from 0 to 1.</p><table><tr><th>Dimension</th>']
 rows += ['<th>'+html.escape(k)+'</th>' for k in v['vectors']];rows.append('</tr>')
 for i,t in enumerate(v['trait_order']):rows.append('<tr><td>'+html.escape(t)+'</td>'+''.join(f'<td>{x[i]:.3f}</td>' for x in v['vectors'].values())+'</tr>')
 rows.append('</table>');path=Path('report.html');path.write_text(path.read_text().replace('</html>',''.join(rows)+'</html>'))
if __name__=='__main__':render_reports()
