import requests,pathlib,json,hashlib
R=pathlib.Path.cwd();assert (R/'.git').is_dir()
api='https://api.github.com/repos/anthropics/evals/commits/main';r=requests.get(api,timeout=60);r.raise_for_status();sha=r.json()['sha']
manifest=[]
for name in ['sycophancy_on_nlp_survey','sycophancy_on_philpapers2020','sycophancy_on_political_typology_quiz']:
 u=f'https://raw.githubusercontent.com/anthropics/evals/{sha}/sycophancy/{name}.jsonl';r=requests.get(u,timeout=90);r.raise_for_status();rows=[json.loads(s) for s in r.text.splitlines() if s.strip()];p=pathlib.Path('datasets/anthropic_verified')/(name+'.jsonl');p.parent.mkdir(exist_ok=True);p.write_bytes(r.content);manifest.append(dict(path=str(p),url=u,revision=sha,bytes=len(r.content),rows=len(rows),sha256=hashlib.sha256(r.content).hexdigest()));print(p,len(rows))
pathlib.Path('notes/anthropic_verified_manifest.json').write_text(json.dumps(manifest,indent=2))
