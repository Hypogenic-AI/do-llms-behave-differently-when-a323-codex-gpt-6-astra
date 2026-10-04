import json,pathlib,re,html,pymupdf as fitz
R=pathlib.Path.cwd();assert(R/'.git').is_dir()
m=[]
for p in sorted(pathlib.Path('notes').glob('*paper_manifest.json')):m+=json.loads(p.read_text())
m.append(json.loads(pathlib.Path('notes/mdpi_paper.json').read_text()))
for x in m:
 x['title']=[html.unescape(t) for t in x['title']]
 d=fitz.open('papers/'+x['id']+'.pdf');s=d[0].get_text();v=re.search(r'arXiv:\s*([\d.]+v\d+)',s);x['version']=v.group(1) if v else 'publisher PDF';x['local_path']='papers/'+x['id']+'.pdf'
pathlib.Path('notes/papers_catalog.json').write_text(json.dumps(m,indent=2))
lines=['# Downloaded papers','',f'{len(m)} verified PDFs. Downloaded 2026-10-04. Metadata and SHA256: [manifest](../notes/papers_catalog.json). Numeric filenames are stable arXiv identifiers. Dates below are first posting years; the version column identifies the PDF actually read.','', '| Paper | Authors | Year | Version | Local PDF |','|---|---|---|---|---|']
for x in m:
 names='; '.join(x['author'][:3])+ ('; et al.' if len(x['author'])>3 else '')
 lines.append(f"| [{x['title'][0]}]({x['url']}) | {names} | {x['date'][0][:4]} | {x['version']} | [{x['id']}]({x['id']}.pdf) |")
lines+=['','All ten user-specified arXiv PDFs were inspected across every three-page chunk, with text-assisted reading and rendered chunk sheets. Devbunova (2603.19426) also received a complete read. Other discovered papers received abstract/methods screening, not a claimed full appendix review. Detailed evidence: [reading log](../notes/reading_log.md) and [extended notes](../notes/extended_reading.md).','', 'PDF chunks and manifests are in `pages/`; duplicate chunk PDFs are ignored by git. Regenerate with `uv run python scripts/prepare_reading.py`. The LessWrong post is saved as HTML/text in `notes/lesswrong.*`; it summarizes an earlier version of 2505.14617 and is not an independent paper.']
pathlib.Path('papers/README.md').write_text('\n'.join(lines)+'\n')
x=json.load(open('notes/search_resolution.json'));keep={1:'2310.13548',2:'2212.09251',3:'2508.02087',7:'2510.08831',8:'2026_machines_prefer_humans',14:'2604.03058',16:'2608.29198',18:'2605.21778',19:'2602.23971',20:'2607.07003',21:'2604.11609',24:'2609.22224'}
lines=['# Search screening log','','39 paper-finder results screened by title and supplied abstract. Retrieval score was not an inclusion decision. Included papers address prompt framing, authorship attribution, internal mechanisms or benchmark definitions needed for the three selected directions. User-specified papers and two directly relevant citation-chased papers are additional to these results.','','| Index | Title | Decision | Reason / file |','|---|---|---|---|']
for i,p in enumerate(x):
 reason=f"Downloaded papers/{keep[i]}.pdf" if i in keep else 'Outside retained scope: user outcomes, conceptual discussion, domain-specific application, training intervention, or general alignment background; no direct prompt authorship/control method needed beyond included work.'
 lines.append(f"| {i} | {p['title']} | {'Include' if i in keep else 'Prune'} | {reason} |")
pathlib.Path('notes/search_screening.md').write_text('\n'.join(lines)+'\n')
