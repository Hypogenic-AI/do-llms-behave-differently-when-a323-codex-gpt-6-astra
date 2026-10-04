import requests, pathlib, concurrent.futures, json, re, hashlib
from pypdf import PdfReader
ROOT=pathlib.Path.cwd(); assert (ROOT/'.git').is_dir()
ids='2607.07003 2604.03058 2605.21778 2604.11609 2608.29198'.split()
def get(i):
 out={'id':i,'url':f'https://arxiv.org/abs/{i}'}
 try:
  r=requests.get(out['url'],timeout=60); r.raise_for_status(); (ROOT/'notes'/f'{i}.html').write_text(r.text)
  for key in ['title','author','date']:
   out[key]=re.findall(r'<meta name="citation_'+key+r'" content="([^"]+)"',r.text)
  r=requests.get(f'https://arxiv.org/pdf/{i}',timeout=90);r.raise_for_status(); assert r.content.startswith(b'%PDF')
  p=ROOT/'papers'/f'{i}.pdf';p.write_bytes(r.content); pdf=PdfReader(p);out.update(pages=len(pdf.pages),bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
  (ROOT/'notes'/f'{i}.txt').write_text('\n'.join(f'\n=== PAGE {j+1} ===\n'+page.extract_text() for j,page in enumerate(pdf.pages)))
 except Exception as e:out['error']=str(e)
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex: results=list(ex.map(get,ids))
(ROOT/'notes/supplementary_paper_manifest.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
