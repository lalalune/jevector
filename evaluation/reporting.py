"""Build aggregate report sections from measured artifacts."""
import json
from pathlib import Path

def read(path):return json.loads(Path(path).read_text())
def sections():
 public=read('runs/evaluation/public-results.json');audit=read('runs/evaluation/profile-audit.json');profiles=read('runs/improvements.json')['results'];controls=read('runs/evaluation/support-controls.json')
 out=[];rows=[]
 for name,r in public['methods'].items():
  m=r['metrics'];n=m['answered_queries'];ci=m['top1_wilson95']
  rows.append([name,f"{round(m['top1']*n)}/{n}",f"{ci[0]:.0%}–{ci[1]:.0%}",f"{m['recall_at_10']:.1%}",f"{m['mrr_at_10']:.3f}",f"{m['ndcg_at_10']:.3f}"])
 out.append(('Public document retrieval',['Method','Top-1','95% interval','Recall@10','MRR@10','nDCG@10'],rows,[
  'SciFact subset: 500 documents and 80 queries. The test split has 45 queries with labeled answers and 15 without labeled answers.',
  'The remaining 20 queries calibrate rejection thresholds. Retrieval scores above use the 45 test queries with labeled answers.',
  'The 64-question science rubric was fixed before loading this evaluation data. Both providers use the same rubric.',
  'BGE-small leads this test. The rubric uses broad categories and may omit paper-specific details.',
  'This is a sampled subset with random distractors, not the official full SciFact benchmark. The public labels were not written for this project.'
 ]))
 rows=[]
 for name,r in public['methods'].items():
  m=r['metrics'];rows.append([name,f"{m['answered_accept_rate']:.1%}",f"{m['unanswerable_false_accept_rate']:.1%}",f"{m['answerability_balanced_accuracy']:.1%}"])
 out.append(('Answer rejection',['Method','Answered accepted ↑','Unanswerable accepted ↓','Balanced accuracy ↑'],rows,[
  'Each method uses one score threshold selected on the calibration split only. The test has 45 answerable and 15 unanswerable queries.',
  'Unanswerable means all labeled relevant documents were removed. Incomplete relevance labels can omit other valid answers.',
  'Rejection remains unreliable for the question vectors. A high similarity score is not a correctness probability.'
 ]))
 rows=[]
 for n in ('clef-64','clef-256','jev-64','jev-256'):
  c=read(f'search_demo/runs/{n}/results.json');m=c['metrics']['asymmetric'];provider,size=n.split('-');rows.append([provider.capitalize(),size,f"{round(m['top1']*20)}/20",f"{m['recall_at_3']:.0%}"])
 for name,path,field in [('BM25',None,'bm25'),('BGE-small','search_demo/runs/bge-small-results.json',None),('BM25 → BGE cosine','search_demo/runs/bge-bm25-results.json',None)]:
  m=read(path)['metrics'] if path else read('search_demo/runs/clef-64/results.json')['metrics'][field];rows.append([name,'384' if path else '—',f"{round(m['top1']*20)}/20",f"{m['recall_at_3']:.0%}"])
 m=controls['Clef one-question classifier']['metrics'];rows.append(['Clef one-question classifier','1 label',f"{round(m['top1']*20)}/20",f"{m['recall_at_3']:.0%}"])
 out.append(('Synthetic support example',['Method','Dimensions','Top-1','Recall@3'],rows,[
  '16 authored documents and 20 authored queries. The examples closely match the task catalog. These are development results.',
  'The one-question classifier selects a task label and retrieves documents with that label. It matches Clef vector top-1 accuracy here.',
  'The 256-dimension vectors do not improve top-1 over 64 dimensions in this example. Use 64 as the starting point.',
  'BM25 → BGE cosine selects 10 lexical candidates. It is not a cross-encoder. The public evaluation above includes RRF and a cross-encoder.'
 ]))
 rows=[]
 for name,r in controls.items():
  if name=='Clef one-question classifier':continue
  m=r['metrics'];rows.append([name,f"{round(m['top1']*20)}/20",f"{m['recall_at_10']:.1%}"])
 out.append(('Support scoring controls',['Method','Top-1','Recall@10'],rows,[
  'These controls use the same saved extraction outputs as the support example. They do not require additional model calls.',
  'Coverage ranks the weighted fraction of requested dimensions supplied by a document. Extra document content does not lower this score.',
  'Coverage does not improve Clef top-1 and reduces Jev top-1 here. It is an optional scoring rule, not a superior default.'
 ]))
 rows=[]
 for n,d in profiles.items():
  provider,size=n.split('-');rows.append([provider.capitalize(),size,f"{round(d['new_live_top1']*32)}/32",f"{round(d['fresh_new_matcher']*16)}/16"])
 for name,label,size in [('BM25 plain','BM25, plain text','—'),('BGE plain, no prefix, chunked','BGE, plain text, no prefix','384'),('BGE plain, query prefix, chunked','BGE, plain text, query prefix','384'),('BM25 + BGE RRF','BM25 + BGE RRF','384')]:
  rows.append([label,size]+[f"{round(audit[group]['variants'][name]['metrics']['top1']*audit[group]['people'])}/{audit[group]['people']}" for group in ('original','additional')])
 out.append(('Synthetic profile retrieval',['Method','Dimensions','Original top-1','Additional top-1'],rows,[
  'Retrieve a second description of the same synthetic person. This does not measure compatibility between different people or personality accuracy.',
  'The original set has 32 people and informed matcher development. The additional set has 16 people and shares the preference catalog.',
  'Both BGE prefix variants are shown. No best-per-set BGE score is selected. Profile fields are preserved in the plain-text conversion.',
  'All 96 legacy BGE inputs fit within 512 tokens. Formatting and prefix choices affected the scores; truncation did not explain the legacy failures.'
 ]))
 rows=[]
 for name in audit['original']['variants']:
  rows.append([name]+[f"{round(audit[g]['variants'][name]['metrics']['top1']*audit[g]['people'])}/{audit[g]['people']}" for g in ('original','additional')])
 out.append(('Profile baseline sensitivity',['Variant','Original top-1','Additional top-1'],rows,[
  'The chunking code uses the pinned BGE tokenizer. It limits each input to 512 tokens, including the prefix and special tokens.',
  'Long texts use 460-token chunks. Normalized chunk vectors are averaged and normalized again. The profile fixtures require no chunks beyond one.'
 ]))
 rows=[]
 for n,d in profiles.items():
  provider,size=n.split('-');rows.append([provider.capitalize(),size,f"{d['fresh_extraction_accuracy']:.2%}",d['fresh_controls']])
 out.append(('Profile extraction checks',['Model','Dimensions','Preference accuracy','Controls'],rows,[
  'These scores use the additional synthetic set. BM25 and BGE are retrieval methods and were not used to extract preferences.',
  'Jev fails the contradictory dating-intent control. Expected: mixed. Received: no. Confirmed user preferences must not be replaced by model inference.'
 ]))
 rows=[]
 latency=read('runs/evaluation/query-latency.json')
 for name,r in public['methods'].items():
  if name=='BGE-small 384':
   da=r['document_embedding'];db=r['query_embedding'];index_tokens=da['input_tokens'];query_tokens=db['input_tokens'];rate=.0202;p50=latency['p50_seconds'];p95=latency['p95_seconds'];measurement='10 sequential queries; local token estimate'
  elif name in ('Clef 64','Jev 64'):
   index_tokens=r['index_input_tokens'];query_tokens=r['query_input_tokens'];rate=.09 if name.startswith('Clef') else .042;p50=r['recorded_query_seconds_p50'];p95=r['recorded_query_seconds_p95'];measurement='80 queries; 6 workers; API token usage'
  else:continue
  rows.append([name,f'{p50:.3f} s',f'{p95:.3f} s',f'${index_tokens*rate/1e6:.4f}',f'${query_tokens/80*1000*rate/1e6:.4f}',measurement])
 out.append(('Model cost and request latency',['Method','Query p50','Query p95','500-doc input cost','1,000-query input cost','Measurement'],rows,[
  'Costs are estimates from input tokens and published rates, not invoices. Failed requests, retries, storage, and index hosting are excluded.',
  'Recorded request durations include network time and any client retries. Concurrency differs as shown. These are not controlled service speed comparisons.',
  'Indexing and query encoding are separate costs. Reducing vector dimensions does not establish lower total search cost.',
  'Rates per million input tokens: BGE-small $0.0202; Clef-flash $0.09; Jev $0.042. Sources are linked in the evaluation instructions.'
 ]))
 cross=public['methods']['RRF + cross-encoder']
 out.append(('Cross-encoder conditions',[],[],[f"RRF selects 20 candidates. BGE-reranker-base scores each query-document pair. Median recorded rerank request: {cross['recorded_query_seconds_p50']:.3f} seconds.",f"{cross['truncated_candidate_inputs']} candidate inputs were shortened to fit a conservative token budget. The budget includes the query.",'This truncation can affect results. Embedding vectors retain all text through chunk pooling.']))
 if Path('runs/evaluation/index-benchmark.json').exists():
  index=read('runs/evaluation/index-benchmark.json');rows=[]
  for r in index['rows']:
   if r['documents']==20000:rows.append([str(r['dimensions']),str(r['ef_search']),f"{r['top10_set_recall']:.1%}",f"{r['hnsw_p50_ms']:.3f}",f"{r['hnsw_p95_ms']:.3f}",f"{r['exact_p50_ms']:.3f}"])
  out.append(('Index mechanics: 20,000 random vectors',['Dimensions','Search effort','Top-10 agreement','HNSW p50 ms','HNSW p95 ms','Exact p50 ms'],rows,[
   '100 seeded random queries, one HNSW thread, M=16, and construction effort 200. This measures index mechanics, not semantic relevance.',
   'Search effort is configurable and no longer grows automatically with corpus size. Higher effort trades speed for exact-search agreement.',
   'The full artifact also includes 500 and 5,000 vectors. Local timings include result ordering and have no production service guarantee.'
  ]))
 from matching.reporting import sections as paired_sections
 from matching.production_reporting import sections as production_sections
 return production_sections()+paired_sections()+out

def markdown(sections):
 lines=[]
 for title,headers,rows,notes in sections:
  lines+=['## '+title,'']
  if headers:lines+=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(row)+' |' for row in rows]+['']
  lines+=notes+['']
 return '\n'.join(lines)
