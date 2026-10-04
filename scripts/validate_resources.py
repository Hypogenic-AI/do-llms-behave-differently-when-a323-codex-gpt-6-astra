"""Read-only validation; writes a report and small, non-sensitive schema examples."""
from pathlib import Path
import json,csv,hashlib,collections
R=Path.cwd();assert (R/'.git').is_dir()
report={}
for p in sorted(Path('datasets').rglob('*')):
 if p.suffix not in ['.csv','.jsonl'] or not p.is_file():continue
 rows=list(csv.DictReader(p.open())) if p.suffix=='.csv' else [json.loads(l) for l in p.open() if l.strip()]
 keys=sorted(set().union(*(x.keys() for x in rows)));info=dict(rows=len(rows),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),schema={k:sorted(set(type(x.get(k)).__name__ for x in rows[:100])) for k in keys},missing_or_empty={k:sum(x.get(k) is None or x.get(k)=='' for x in rows) for k in keys},duplicate_full_rows=len(rows)-len(set(json.dumps(x,sort_keys=True) for x in rows)))
 if 'question' in keys:info['unique_questions']=len(set(x['question'] for x in rows));info['bad_answer_pairs']=sum(x['answer_matching_behavior']==x['answer_not_matching_behavior'] for x in rows)
 if 'conversation' in keys:
  info['language_counts']=dict(collections.Counter(x.get('language') for x in rows));info['empty_conversation']=sum(not x.get('conversation') for x in rows);info['first_turn_user']=sum(bool(x.get('conversation')) and x['conversation'][0].get('role')=='user' for x in rows);info['unique_conversation_hashes']=len(set(x.get('conversation_hash') for x in rows));info['viewer_indices_contiguous']=[x['source_row_idx'] for x in rows]==list(range(1000))
 report[str(p)]=info
Path('notes/dataset_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:{kk:v[kk] for kk in ['rows','duplicate_full_rows']+(['language_counts','empty_conversation','first_turn_user','unique_conversation_hashes'] if 'wildchat' in k else [])} for k,v in report.items()},indent=2))
Path('datasets/samples').mkdir(exist_ok=True)
benign=list(csv.DictReader(open('datasets/jbb/data/benign-behaviors.csv')))[:3];Path('datasets/samples/jbb_benign.json').write_text(json.dumps(benign,indent=2))
for p in Path('datasets/anthropic_verified').glob('*.jsonl'):
 Path('datasets/samples',p.stem+'.json').write_text(json.dumps(json.loads(p.open().readline()),indent=2))
# WildChat schema sample avoids distributing conversations, IPs or location metadata.
p=Path('datasets/wildchat/first_1000.jsonl');x=json.loads(p.open().readline());Path('datasets/samples/wildchat_schema.json').write_text(json.dumps({k:type(v).__name__ for k,v in x.items()},indent=2))
