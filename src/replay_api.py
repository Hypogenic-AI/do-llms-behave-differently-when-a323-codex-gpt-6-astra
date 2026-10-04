"""Uncached exact-request repeat on 24 preselected successful API cells."""
import json,random,time,requests,os
from pathlib import Path
from common import dump,parallel,SEED,content

def replay(r):
    old=json.load(open('results/raw/'+r['key']+'.json'));path=Path('results/raw/replay_'+r['key']+'.json')
    if path.exists():new=json.load(open(path))
    else:
        response=requests.post('https://openrouter.ai/api/v1/chat/completions',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_KEY']},json=old['request'],timeout=180)
        response.raise_for_status();new={'key':'replay_'+r['key'],'tag':'api_replay','request':old['request'],'response':response.json(),'timestamp':time.time()};dump(path,new)
    return {**r,'text':content(new),'key':new['key'],'original_text':r['text'],'exact_text_match':content(new)==r['text']}
if __name__=='__main__':
    rows=json.load(open('results/api_outputs_long.json'));rng=random.Random(SEED);sample=[]
    for m in sorted({r['model'] for r in rows}):
        for d in ['harmful','benign','sycophancy']:
            rr=[r for r in rows if r['model']==m and r['domain']==d and r['kind']=='behavior' and r['label']=='none' and not r.get('error')]
            sample.extend(rng.sample(rr,4))
    out=parallel(replay,sample,workers=8);dump('results/api_replay.json',out)
    dump('results/api_replay_summary.json',{'n':len(out),'exact_text_match':sum(r['exact_text_match'] for r in out)})
