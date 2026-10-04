"""Cached OpenRouter calls with raw requests, responses, usage and bounded retries."""
import os, json, time, hashlib, threading, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
ROOT=Path(__file__).resolve().parents[1]
assert Path.cwd()==ROOT, 'Run from workspace root'
SEED=42
GEM='google/gemini-3.1-flash-lite'
QWEN='qwen/qwen3.5-9b'
WRITER='qwen/qwen3.5-397b-a17b'
SYSTEM='You are a helpful assistant.'

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False))

def parse_json(s):
    s=re.sub(r'^```(?:json)?\s*|\s*```$','',s.strip())
    return json.loads(s[s.find('{'):s.rfind('}')+1])

def call(model,prompt,system=SYSTEM,max_tokens=256,tag='call'):
    payload={'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':prompt}],
             'temperature':0,'top_p':1,'max_tokens':max_tokens,'seed':SEED,
             'reasoning':({'effort':'minimal'} if model==GEM else {'enabled':False})}
    key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    path=ROOT/'results/raw'/f'{key}.json'
    if path.exists(): return json.loads(path.read_text())
    for attempt in range(4):
        start=time.time()
        try:
            r=requests.post('https://openrouter.ai/api/v1/chat/completions',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_KEY']},json=payload,timeout=150)
            r.raise_for_status(); data=r.json()
            if 'choices' not in data: raise ValueError(str(data)[:500])
            out={'key':key,'tag':tag,'request':payload,'response':data,'seconds':time.time()-start,'timestamp':time.time()}
            dump(path,out);return out
        except Exception as e:
            if attempt==3:
                out={'key':key,'tag':tag,'request':payload,'error':str(e),'timestamp':time.time()};dump(path,out);return out
            time.sleep(2**attempt)

def content(r):
    return r.get('response',{}).get('choices',[{'message':{}}])[0]['message'].get('content') or ''

def parallel(fn,items,workers=16):
    out=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fs=[ex.submit(fn,x) for x in items]
        for i,f in enumerate(as_completed(fs)):
            out.append(f.result())
            if (i+1)%25==0: print(f'{fn.__name__}: {i+1}/{len(items)}',flush=True)
    return out

def state(note):
    p=ROOT/'STATE.md';s=p.read_text();a='<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->';b='<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->'
    p.write_text(s[:s.index(a)+len(a)]+'\n'+note+'\n'+s[s.index(b):])
