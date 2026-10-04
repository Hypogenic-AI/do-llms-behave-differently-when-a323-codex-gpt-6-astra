from pathlib import Path
import fitz,subprocess,json
assert Path('.git').is_dir()
for p in sorted(Path('papers').glob('*.pdf')):
 subprocess.run(['.venv/bin/python','.codex/skills/paper-finder/scripts/pdf_chunker.py',str(p),'--pages-per-chunk','3'],check=True,stdout=subprocess.DEVNULL)
 d=fitz.open(p)
 # Text from each page supplements visual inspection without missing appendix pages.
 Path('notes',p.stem+'.txt').write_text('\n'.join(f'\n=== PAGE {i+1} ===\n'+page.get_text() for i,page in enumerate(d)))
 print(p.stem,len(d))
