"""Uniform larger response budget to repair first-pass truncation, keeping originals."""
import json,random
from common import *

def extend(t):
    if t['kind']=='perception' or t.get('error'):return t
    r=call(t['model'],t['prompt'],max_tokens=1536,tag='behavior_extended')
    return {**t,'initial_key':t['key'],'key':r['key'],'text':content(r),'error':r.get('error'),'finish_reason':r.get('response',{}).get('choices',[{}])[0].get('finish_reason'),'max_tokens':1536}
if __name__=='__main__':
    rows=[]
    for name in ['api','format']:
        rows.extend([{**r,'dataset':name} for r in json.load(open(f'results/{name}_outputs.json'))])
    random.Random(SEED).shuffle(rows);out=parallel(extend,rows,workers=24)
    for name in ['api','format']:dump(f'results/{name}_outputs_long.json',[r for r in out if r['dataset']==name])
