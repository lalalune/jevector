"""Aggregate paired-profile benchmarks without disclosing individual profiles in the viewer."""
import json
from pathlib import Path

def sections():
 path=Path('runs/matching/results.json')
 if not path.exists():return []
 result=json.loads(path.read_text())
 from jevector import digest
 from matching.schema import SCHEMA_HASH
 data=json.loads(Path('matching/data.json').read_text())
 if result['data_hash']!=digest(data) or result['schema_hash']!=SCHEMA_HASH:raise ValueError('Matching results do not match current data and schema')
 methods=result['methods'];rows=[]
 primary=['Clef 64','Clef 256','Jev 64','Jev 256','BM25 reciprocal','BGE reciprocal, no prefix','BGE reciprocal, query prefix','BM25 + BGE RRF','BM25 top-10 + BGE rerank','Template parser control']
 for name in primary:
  if name not in methods:continue
  m=methods[name]['metrics'];n=m['paired_queries'];rows.append([name,f"{round(m['partner_top1']*n)}/{n}",f"{m['partner_recall5']:.1%}",f"{m['both_direction_pairs']}/{m['pairs']}",f"{m['hard_violation_count']}/{m['returned_count']}",f"{m['paired_abstentions']}/{n}",f"{m['no_match_rejections']}/{m['no_match_queries']}"])
 notes=['164 synthetic profiles: 16 intended pairs of different people, 96 hard distractors, 32 valid lower-ranked alternatives, and four no-match queries.',
  'Each of the 32 paired people searches the other 163 profiles. Each has exactly two source-valid candidates: the partner and one lower-ranked alternative. The matcher checks both people’s mandatory conditions.',
  'This is a development fixture. Prompts and the input contract were corrected after pilot runs. It is not a held-out evaluation.',
  'Pair assignments and source attributes are evaluator-only. Models receive the person’s own description and stated partner preferences.',
  'Hard violations count incompatible first results among returned first results. Paired abstentions count queries where a valid partner exists but no result is returned.',
  'The template-parser control recognizes the fixture wording without a model. This measures fixture simplicity; it is not a general free-text parser.',
  'The profiles are generated from explicit matching specifications. This tests rule-based compatibility, not real-world relationship outcomes.',
  'BM25 and BGE average two directional scores. These unfiltered reference methods do not enforce mandatory conditions or reject no-match queries.']
 out=[('Historical reciprocal profile matching',['Method','Partner top-1','Partner recall@5','Both ways','Hard violations ↓','Paired abstentions ↓','No-match rejected ↑'],rows,notes)]
 rows=[]
 for name,method in methods.items():
  if 'constraints + BGE' not in name:continue
  m=method['metrics'];rows.append([name,f"{round(m['partner_top1']*32)}/32",f"{m['partner_recall5']:.1%}",f"{m['hard_violation_count']}/{m['returned_count']}"])
 out.append(('Extracted constraints with BGE ranking',['Method','Partner top-1','Partner recall@5','Hard violations ↓'],rows,[
  'The same model-extracted reciprocal conditions filter candidates. BGE then ranks eligible profiles.',
  'This separates the contribution of explicit constraints from the choice of ranking score. No ground-truth attributes are used for filtering.'
 ]))
 analysis=Path('runs/matching/analysis.json')
 if analysis.exists():
  report=json.loads(analysis.read_text())
  if report['data_hash']!=result['data_hash'] or report.get('schema_hash')!=SCHEMA_HASH:raise ValueError('Matching analysis is stale')
  diagnostics=report['methods'];rows=[]
  for name,record in diagnostics.items():
   for channel,m in record['extraction'].items():
    rows.append([name,channel,f"{m['positive_precision']:.1%}",f"{m['positive_recall']:.1%}",f"{m['known_accuracy']:.1%}",f"{m['unknown_accuracy']:.1%}" if m['unknown_accuracy'] is not None else '—'])
  out.append(('Reciprocal extraction checks',['Model','Channel','Yes precision','Yes recall','Known accuracy','Unknown accuracy'],rows,[
   'Self attributes, optional wishes, requirements, and exclusions are scored separately against the source specification.',
   'Positive precision and recall expose missed or invented conditions. Accuracy alone can hide errors when most conditions are absent.',
   'The 64-value schema uses 16 attributes across four channels. The 256-value schema uses 64 attributes across those channels.',
   'A single uncertain mandatory condition can reject a valid pair. High average attribute accuracy does not guarantee high match recall.',
   'All mandatory fixture conditions use the core attributes. Extra attributes affect optional ranking. This schema differs from the older profile-retrieval schema.'
  ]))
 rows=[]
 for name in primary[:4]:
  if name not in methods:continue
  m=methods[name];rate=.09 if name.startswith('Clef') else .042
  rows.append([name,f"{m['input_tokens']:,}",f"${m['input_tokens']*rate/1e6:.4f}",f"{m['recorded_profile_seconds_p50']:.3f} s"])
 out.append(('Reciprocal extraction cost',['Model','Input tokens','164-profile input cost','Profile p50'],rows,[
  'Estimates use API input-token counts and published input rates. They exclude failed requests, retries, storage, and hosting.',
  'Profiles are extracted once and reused for both sides of every match. Matching existing vectors requires no model call.',
  'Profiles run concurrently. Each profile uses four sequential channel requests at either size; its duration is their sum.'
 ]))
 return out
