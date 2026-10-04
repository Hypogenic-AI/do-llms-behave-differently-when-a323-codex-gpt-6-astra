"""Replay recorded URLs, verify SHA256 before writing. Run from workspace root."""
import json,pathlib,requests,hashlib
R=pathlib.Path.cwd();assert (R/'.git').is_dir();print(R)
for manifest in ['notes/dataset_manifest.json','notes/anthropic_verified_manifest.json']:
 for x in json.load(open(manifest)):
  p=pathlib.Path(x['path'])
  if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']:continue
  if 'wildchat' in str(p):
   rows=[]
   for offset in range(0,1000,100):
    r=requests.get('https://datasets-server.huggingface.co/rows',params=dict(dataset='allenai/WildChat-1M',config='default',split='train',offset=offset,length=100),timeout=90);r.raise_for_status();rows.extend(dict(source_row_idx=z['row_idx'],**z['row']) for z in r.json()['rows'])
   data=(''.join(json.dumps(z,ensure_ascii=False)+'\n' for z in rows)).encode()
  else:
   r=requests.get(x['url'],timeout=120);r.raise_for_status();data=r.content
  if hashlib.sha256(data).hexdigest()!=x['sha256']:raise RuntimeError(f'Snapshot mismatch: {p}; file not overwritten. WildChat viewer is unpinned; use local archive or record a new snapshot explicitly.')
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);print(p)
print('All recorded dataset files verified.')
