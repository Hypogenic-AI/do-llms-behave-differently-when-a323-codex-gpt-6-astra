"""Pinned benchmark downloads and a deterministic 1,000-row WildChat viewer subset."""
import requests,json,pathlib,hashlib,csv,io,time
R=pathlib.Path.cwd();assert (R/'.git').is_dir()
manifest=[]
def save(url,p):
 for attempt in range(3):
  try:
   r=requests.get(url,timeout=90);r.raise_for_status();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content);return r
  except requests.RequestException:
   if attempt==2:raise
   time.sleep(2)
def record(p,url,n=None):manifest.append(dict(path=str(p),url=url,bytes=p.stat().st_size,rows=n,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
for ds,folder in [('JailbreakBench/JBB-Behaviors','jbb'),('Anthropic/model-written-evals','anthropic')]:
 meta=json.loads((R/'notes'/ (ds.replace('/','_')+'.json')).read_text()); sha=meta['sha']
 files=[x['rfilename'] for x in meta['siblings'] if x['rfilename'].endswith('.csv')] if folder=='jbb' else [x['rfilename'] for x in meta['siblings'] if x['rfilename'].startswith('sycophancy/') and x['rfilename'].endswith('.jsonl')]
 files+=['README.md']
 for f in files:
  p=pathlib.Path('datasets')/folder/f;url=f'https://huggingface.co/datasets/{ds}/resolve/{sha}/{f}';res=save(url,p)
  n=len(list(csv.DictReader(io.StringIO(res.text)))) if f.endswith('.csv') else len([json.loads(l) for l in res.text.splitlines() if l.strip()]) if f.endswith('.jsonl') else None
  record(p,url,n);print(p,n,flush=True)
rows=[]
for offset in range(0,1000,100):
 url=f'https://datasets-server.huggingface.co/rows?dataset=allenai/WildChat-1M&config=default&split=train&offset={offset}&length=100'
 r=requests.get(url,timeout=90);r.raise_for_status();d=r.json();rows.extend(dict(source_row_idx=x['row_idx'],**x['row']) for x in d['rows'])
p=pathlib.Path('datasets/wildchat/first_1000.jsonl');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));record(p,'https://datasets-server.huggingface.co/rows (offsets 0..900, default/train, length 100; unpinned viewer snapshot)',len(rows))
pathlib.Path('notes/dataset_manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',len(rows))
