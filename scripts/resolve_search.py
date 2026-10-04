import requests,json,pathlib,concurrent.futures,re
from pypdf import PdfReader
root=pathlib.Path.cwd();assert (root/'.git').is_dir()
ps=[json.loads(l) for l in sorted(pathlib.Path('paper_search_results').glob('*.jsonl'))[-1].read_text().splitlines()]
def run(t):
 i,d=t;u=d['url'];pid=u.split('/')[-1];o=dict(index=i,**d)
 try:
  r=requests.get(f'https://api.semanticscholar.org/graph/v1/paper/{pid}',params={'fields':'title,externalIds,openAccessPdf,url,authors,year'},timeout=40);r.raise_for_status();m=r.json();o['metadata']=m
  aid=m.get('externalIds',{}).get('ArXiv');pdfurl=f'https://arxiv.org/pdf/{aid}' if aid else (m.get('openAccessPdf') or {}).get('url')
  if not pdfurl: o['download_error']='No open PDF in Semantic Scholar metadata';return o
  r=requests.get(pdfurl,timeout=70);r.raise_for_status();assert r.content.startswith(b'%PDF'), 'Not PDF'
  p=pathlib.Path('papers')/f'search_{i:02d}_{aid or "paper"}.pdf';p.write_bytes(r.content);reader=PdfReader(p);o.update(file=str(p),pages=len(reader.pages),pdf_url=pdfurl)
 except Exception as e:o['download_error']=str(e)
 return o
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 out=list(pool.map(run,enumerate(ps)))
pathlib.Path('notes/search_resolution.json').write_text(json.dumps(out,indent=2));print('\n'.join(f'{d["index"]}: {d.get("file",d.get("download_error"))}' for d in out))
