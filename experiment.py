"""64-dimensional synthetic personality retrieval experiment; Python 3.11+, stdlib only."""
import argparse, hashlib, json, math, os, pathlib, random, time, tomllib, urllib.request, urllib.error
ROOT = pathlib.Path(__file__).resolve().parent
MODEL = '@cf/cloudflare/clef-flash'
# Eight broad groups, eight observable qualities each. This is an experimental rubric.
GROUPS = {
'social': [('sociability','seeks frequent social interaction'),('assertiveness','states needs and opinions directly'),('expressiveness','openly communicates feelings'),('warmth','makes others feel welcomed'),('humor','uses playful humor'),('leadership','takes responsibility for guiding groups'),('social_confidence','feels at ease with unfamiliar people'),('attention_seeking','seeks public notice')],
'cooperation': [('empathy','notices and understands others feelings'),('compassion','acts to relieve others distress'),('generosity','shares time or resources'),('trust','expects good intentions from others'),('cooperativeness','works toward shared goals'),('forgiveness','lets go of personal grievances'),('humility','acknowledges limitations without self-promotion'),('politeness','shows courtesy during disagreement')],
'execution': [('organization','keeps materials and tasks ordered'),('reliability','follows through on commitments'),('self_discipline','works despite distractions'),('diligence','puts sustained care into work'),('planning','prepares before acting'),('punctuality','arrives and delivers on time'),('attention_to_detail','checks small details carefully'),('persistence','continues despite setbacks')],
'curiosity': [('curiosity','actively seeks new understanding'),('creativity','produces novel ideas'),('imagination','envisions possibilities beyond current facts'),('open_mindedness','seriously considers unfamiliar viewpoints'),('aesthetic_appreciation','notices and values beauty or art'),('intellectual_engagement','enjoys complex ideas'),('experimentation','tries alternatives to learn what works'),('reflectiveness','examines own experiences and assumptions')],
'regulation': [('emotional_stability','maintains emotional balance under stress'),('patience','tolerates delays without irritation'),('resilience','recovers after setbacks'),('optimism','expects favorable future outcomes'),('self_confidence','trusts own ability'),('stress_tolerance','functions effectively under pressure'),('impulse_control','pauses before acting on urges'),('adaptability','adjusts when circumstances change')],
'drive': [('ambition','seeks demanding achievements'),('initiative','starts useful action without prompting'),('competitiveness','wants to outperform others'),('decisiveness','commits to choices promptly'),('independence','prefers making own decisions'),('risk_tolerance','accepts uncertain outcomes for potential benefit'),('energy','sustains an active pace'),('adventurousness','seeks unfamiliar experiences')],
'values': [('honesty','communicates truthfully even at personal cost'),('fairness','applies standards equitably'),('integrity','acts consistently with stated principles'),('accountability','accepts responsibility for mistakes'),('loyalty','supports established relationships through difficulty'),('respect_for_boundaries','honors others limits and consent'),('civic_mindedness','contributes to shared community welfare'),('environmental_care','considers environmental effects of choices')],
'style': [('flexibility','readily changes preferred methods'),('spontaneity','enjoys unplanned action'),('practicality','prioritizes workable concrete solutions'),('analytical_thinking','breaks problems into evidence-based parts'),('cautiousness','checks potential downsides before acting'),('frugality','uses money and resources sparingly'),('order_preference','prefers predictable structured surroundings'),('conflict_tolerance','engages constructively with disagreement')]
}
TRAITS = [{'id':k,'group':g,'definition':d} for g,ts in GROUPS.items() for k,d in ts]
assert len(TRAITS)==64 and len({t['id'] for t in TRAITS})==64

def questions():
 return {t['id']:{'type':'score','instructions':f"Based only on described behavior, rate how consistently this person {t['definition']}. Do not infer from identity or occupation. Use Mixed or insufficient evidence when unspecified.",'criteria':['Consistently contrary behavior','Usually contrary behavior','Mixed or insufficient evidence','Usually demonstrates this quality','Consistently demonstrates this quality']} for t in TRAITS}

def token():
 for key in ('CLOUDFLARE_API_TOKEN','CLOUDFLARE_AUTH_TOKEN'):
  if os.environ.get(key): return os.environ[key]
 for p in [pathlib.Path.home()/'Library/Preferences/.wrangler/config/default.toml',pathlib.Path.home()/'.wrangler/config/default.toml']:
  if p.exists(): return tomllib.loads(p.read_text())['oauth_token']
 raise RuntimeError('No Cloudflare API token or Wrangler login found')

def api(path,body=None):
 req=urllib.request.Request('https://api.cloudflare.com/client/v4/'+path,data=json.dumps(body).encode() if body else None,headers={'Authorization':'Bearer '+token(),'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=180) as r: data=json.load(r)
 except urllib.error.HTTPError as e:
  raise RuntimeError(f'Cloudflare HTTP {e.code}: {e.read().decode()[:1500]}') from None
 if not data.get('success',True): raise RuntimeError(str(data.get('errors')))
 return data.get('result',data)

def distance(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a))
def evaluate(vectors,people):
 ids=[p['id'] for p in people]; a=[vectors[i+'_a'] for i in ids]; b=[vectors[i+'_b'] for i in ids]
 matrix=[[distance(x,y) for y in a] for x in b]
 rows=[]
 for i,ds in enumerate(matrix):
  order=sorted(range(len(ids)),key=lambda j:ds[j]); rank=order.index(i)+1
  rows.append({'person':ids[i],'matched':ids[order[0]],'rank':rank,'own_distance':ds[i],'nearest_other_distance':min(v for j,v in enumerate(ds) if j!=i),'top3':[ids[j] for j in order[:3]]})
 observed=sum(r['rank']==1 for r in rows)/len(rows)
 rng=random.Random(42); exceed=0
 for _ in range(10000):
  labels=list(range(len(ids)));rng.shuffle(labels)
  score=sum(min(range(len(ids)),key=lambda j:matrix[i][j])==labels[i] for i in range(len(ids)))/len(ids)
  exceed+=score>=observed
 return {'top1_accuracy':observed,'top3_accuracy':sum(r['rank']<=3 for r in rows)/len(rows),'mean_reciprocal_rank':sum(1/r['rank'] for r in rows)/len(rows),'chance_top1':1/len(rows),'label_permutation_p':(exceed+1)/10001,'matches':rows,'distance_matrix_query_b_to_gallery_a':matrix}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--account');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
 (ROOT/'traits.json').write_text(json.dumps(TRAITS,indent=2)+'\n');(ROOT/'questions.json').write_text(json.dumps(questions(),indent=2)+'\n')
 people=json.loads((ROOT/'people.json').read_text());assert len({p['id'] for p in people})==len(people)
 if args.prepare_only: print(f'Prepared {len(TRAITS)} traits and {len(people)} paired synthetic people');return
 account=args.account or os.environ.get('CLOUDFLARE_ACCOUNT_ID')
 if not account:
  accounts=api('accounts')
  if len(accounts)!=1: raise RuntimeError('Select account with --account; available: '+str([{'id':a['id'],'name':a['name']} for a in accounts]))
  account=accounts[0]['id']
 out=ROOT/'results';out.mkdir(exist_ok=True);vectors={};timings=[]
 for p in people:
  for view in ('a','b'):
   key=p['id']+'_'+view;body={'model':'clef-flash','state':p[view],'questions':questions()};digest=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();path=out/(key+'.json')
   if path.exists():
    record=json.loads(path.read_text())
    if record['request_sha256']!=digest: raise RuntimeError(f'Stale cache: {path}')
   else:
    start=time.monotonic();result=api(f'accounts/{account}/ai/run/{MODEL}',body)
    record={'model':MODEL,'request_sha256':digest,'elapsed_seconds':time.monotonic()-start,'response':result};path.write_text(json.dumps(record,indent=2)+'\n')
   answers=record['response']['answers'];assert set(answers)=={t['id'] for t in TRAITS}
   vector=[float(answers[t['id']]['score'])/4 for t in TRAITS];assert all(math.isfinite(v) and 0<=v<=1 for v in vector)
   vectors[key]=vector;timings.append(record['elapsed_seconds']);print(key,'scored',round(record['elapsed_seconds'],2),'seconds',flush=True)
 (out/'vectors.json').write_text(json.dumps({'trait_order':[t['id'] for t in TRAITS],'vectors':vectors},indent=2)+'\n')
 report=evaluate(vectors,people);report['mean_request_seconds']=sum(timings)/len(timings);report['model']=MODEL;report['people']=len(people);report['requests']=len(vectors)
 (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__': main()
