"""Generate benchmark documentation and the aggregate viewer from saved measurements."""
import html
from pathlib import Path
from evaluation.reporting import sections,markdown

def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def main():
 data=sections();report='# JEVECTOR / PROFILE MATCHING\n\nExplicit profile values, reciprocal requirements, and measured matching results. Results use saved API responses.\n\n'+markdown(data)
 Path('search_demo/RESULTS.md').write_text(report)
 Path('evaluation/REPORT.md').write_text(report)
 readme=Path('README.md');text=readme.read_text();start='<!-- benchmarks:start -->';end='<!-- benchmarks:end -->'
 if start in text:
  a=text.index(start)+len(start);b=text.index(end);text=text[:a]+'\n\n'+markdown([section for section in data if section[0] in ('Retrieval comparison','Complete pipeline comparison')])+'\n'+text[b:];readme.write_text(text)
 page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>JEVECTOR / PROFILE MATCHING</title><style>
body{font:16px system-ui;color:#172b42;background:#f4f6f9;max-width:1200px;margin:32px auto;padding:0 24px}h1{font-size:28px;letter-spacing:.04em}h2{font-size:21px}p,dd{line-height:1.5}.panel{background:white;border:1px solid #d9e1e9;border-radius:8px;padding:22px;margin:20px 0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:12px 9px;text-align:left;border-bottom:1px solid #dde4ea}th{font-size:13px;color:#52677b}td:not(:first-child){white-space:nowrap}dt{font-weight:650;margin-top:14px}dd{margin:5px 0}.muted{color:#52677b;font-size:14px}a{color:#245b93}summary{cursor:pointer;font-weight:600}</style></head><body>
<h1>JEVECTOR / PROFILE MATCHING</h1><p>Decision-model vectors → HNSW candidates → reciprocal checks → ranked profiles.</p>
<p class="muted">Compare Clef, Jev, BGE-small 384, BM25, and BM25 + BGE-small on the same profiles and queries. Retrieval scores and rule-filtered results are separate.</p>'''
 for i,(title,headers,rows,notes) in enumerate(data):
  section='<section class="panel"><h2>'+html.escape(title)+'</h2>'
  if headers:section+=table(headers,rows)
  section+=''.join('<p class="muted">'+html.escape(note)+'</p>' for note in notes)+'</section>'
  if title not in ('Retrieval comparison','Complete pipeline comparison'):
   section='<details class="panel"><summary>'+html.escape(title)+'</summary>'+section+'</details>'
  page+=section
 keys=[('Partner top-1','The intended different person is ranked first. Each table states the evaluated query count.'),('Both ways','Both members of a pair retrieve each other first.'),('Hard violations','Current qualification counts all returned candidates that fail a source condition. The historical table counts only first results.'),('Paired abstentions','No result returned despite an intended partner being present.'),('No-match rejected','No candidate returned for a query with no source-eligible candidate. Approximate search alone cannot prove that none exists.'),('Question vectors','Each value represents a specified attribute or partner condition. Profile matching checks mandatory conditions in both directions, then ranks optional preference coverage.'),('BM25','Rank documents using word matches and frequency statistics.'),('BGE-small','Use 384-dimensional embeddings and cosine similarity. Long inputs are split into token-bounded chunks.'),('BM25 + BGE rerank','Select the ten highest BM25 candidates, then rank them by BGE similarity. This does not enforce mandatory profile conditions.'),('RRF','Combine ranking positions from BM25 and BGE. The fixed reciprocal-rank constant is 60.'),('Cross-encoder','Score query-document pairs with BGE-reranker-base after RRF selects 20 candidates.'),('Top-1','Queries with a labeled relevant document first. Public retrieval metrics exclude queries with no labeled answer.'),('Recall@10','Mean fraction of labeled relevant documents present in the first ten results.'),('MRR@10','Mean reciprocal rank of the first relevant result. Use zero after position ten.'),('nDCG@10','Discount relevant results by their position, then divide by the ideal score.'),('95% interval','Wilson interval for top-1 proportions. It reflects sampling uncertainty, not dataset or model bias.')]
 page+='<section class="panel"><h2>Method and metric key</h2><dl>'+''.join('<dt>'+html.escape(k)+'</dt><dd>'+html.escape(v)+'</dd>' for k,v in keys)+'</dl></section>'
 page+='<p><a href="README.md">Run instructions</a> · <a href="RESULTS.md">Full report</a></p></body></html>'
 Path('search_demo/index.html').write_text(page)
 print('Generated viewer, README benchmarks, and full reports from measured artifacts.')
if __name__=='__main__':main()
