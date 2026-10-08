"""Additional live controls and report; run after experiment.py."""
import argparse, collections, html, json, math, os, re, statistics, time
from experiment import ROOT, TRAITS, MODEL, api, questions, distance, evaluate
parser=argparse.ArgumentParser(); parser.add_argument('--account', default=os.environ.get('CLOUDFLARE_ACCOUNT_ID')); args=parser.parse_args()
if not args.account: parser.error('--account or CLOUDFLARE_ACCOUNT_ID is required')
ACCOUNT=args.account
people=json.loads((ROOT/'people.json').read_text());out=ROOT/'results'
v=json.loads((out/'vectors.json').read_text())['vectors'];metrics=json.loads((out/'metrics.json').read_text())
# Local text baseline: token TF-IDF cosine retrieval (no external embeddings).
stop=set('a an the and or to of in on for with is are was be by as at from it that this when then but than into own others person'.split())
def words(s):return [w for w in re.findall(r'[a-z]+',s.lower()) if w not in stop]
docs=[words(p[k]) for p in people for k in ('a','b')];df=collections.Counter(w for doc in docs for w in set(doc))
def tfidf(s):
 c=collections.Counter(words(s));r={w:n*(math.log((1+len(docs))/(1+df[w]))+1) for w,n in c.items()};norm=math.sqrt(sum(x*x for x in r.values()));return {w:x/norm for w,x in r.items()}
a=[tfidf(p['a']) for p in people];b=[tfidf(p['b']) for p in people]
ranks=[]
for i,q in enumerate(b):
 scores=[sum(x*g.get(w,0) for w,x in q.items()) for g in a];ranks.append(sorted(range(len(a)),key=lambda j:-scores[j]).index(i)+1)
metrics['lexical_tfidf_baseline']={'top1_accuracy':sum(r==1 for r in ranks)/len(ranks),'ranks':ranks,'idf_fit':'All synthetic texts; descriptive baseline, not trained evaluation'}
controls={}
for name,state,qs in [('repeat',people[0]['a'],questions()),('reverse_question_order',people[0]['a'],dict(reversed(list(questions().items())))),('no_evidence','No behavioral information about this person is available.',questions())]:
 path=out/(name+'.json')
 if path.exists():record=json.loads(path.read_text())
 else:
  t=time.monotonic();record={'response':api(f'accounts/{ACCOUNT}/ai/run/{MODEL}',{'model':'clef-flash','state':state,'questions':qs}),'elapsed_seconds':time.monotonic()-t};path.write_text(json.dumps(record,indent=2)+'\n')
 cv=[record['response']['answers'][t['id']]['score']/4 for t in TRAITS]
 controls[name]={'rms_difference_from_original':distance(cv,v['p01_a'])} if name!='no_evidence' else {'mean_absolute_deviation_from_midpoint':statistics.mean(abs(x-.5) for x in cv),'min_score':min(cv),'max_score':max(cv)}
 print(name,controls[name],flush=True)
metrics['controls']=controls
files=[p for p in out.glob('*.json') if p.name not in ('vectors.json','metrics.json')]
tokens=sum(json.loads(p.read_text()).get('response',{}).get('usage',{}).get('input_tokens',0) for p in files)
metrics['total_input_tokens_including_controls']=tokens;metrics['estimated_model_cost_usd']=tokens*.09/1e6
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
from documentation import render_reports
render_reports()
print('Updated initial personality report.')
