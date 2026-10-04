"""Semantically exact format control: identical request and wrapper words.
Added after rewrite-audit failure, before analyzing behavior. This is explicitly
format/case sensitivity, not a general human-versus-machine style contrast.
"""
import json,re,random
from common import *
from api_experiment import run_task,MODELS

def variants(q):
    return {'human':'please help with this request: '+q+'\nthanks.',
            'ai':'**Please help with this request:**\n\n'+q+'\n\n**Thanks.**'}
if __name__=='__main__':
    rows=[r for r in json.load(open('results/processed/base_requests.json')) if r['domain']!='calibration'];tasks=[]
    for r in rows:
        vv=variants(r['original'])
        # Same lexical content, including case-insensitive wrapper words; no request edits.
        assert re.findall(r'\w+',vv['human'].lower())==re.findall(r'\w+',vv['ai'].lower())
        assert all(r['original'] in p for p in vv.values())
        for m in MODELS:
            for s,p in vv.items():
                for kind in ['behavior','perception']:
                    tasks.append({'id':r['id'],'domain':r['domain'],'model':m,'style':s,'label':'none','kind':kind,'prompt':p,'accepted':True,'experiment':'format_control'})
    random.Random(SEED).shuffle(tasks);dump('results/format_tasks.json',tasks)
    out=parallel(run_task,tasks,workers=24);dump('results/format_outputs.json',out)
