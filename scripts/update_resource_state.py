from pathlib import Path
import sys
p=Path('STATE.md');s=p.read_text();a='<!-- NEURICO_AGENT_NOTES_START:resource_finder -->';b='<!-- NEURICO_AGENT_NOTES_END:resource_finder -->';before,rest=s.split(a,1);_,after=rest.split(b,1);p.write_text(before+a+'\n'+sys.stdin.read().strip()+'\n'+b+after)
