"""Independent-model coding agreement on a seeded stratified sample; not human validation."""
import json,random
from collections import defaultdict
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from common import *
from judge import judge
import sys
expanded='--long' in sys.argv
rows=json.load(open('results/api_scored_long.json' if expanded else 'results/api_scored.json'))
groups=defaultdict(list)
for r in rows:
    if r['kind']=='behavior' and r['label']=='none':groups[(r['model'],r['domain'])].append(r)
rng=random.Random(SEED);sample=[]
for key,rr in sorted(groups.items()):
    for r in rng.sample(rr,min(15,len(rr))):
        sample.append({**r,'score_first':r['score'],'_judge_override':GEM if r['judge_model']==WRITER else WRITER})
out=parallel(judge,sample);dump('results/judge_reliability_raw_long.json' if expanded else 'results/judge_reliability_raw.json',out)
metrics=[]
for domain in ['harmful','benign','sycophancy']:
    rr=[r for r in out if r['domain']==domain and 'score_error' not in r];k='alignment' if domain=='sycophancy' else 'refusal'
    a=[str(r['score_first'].get(k)) for r in rr];b=[str(r['score'].get(k)) for r in rr]
    metrics.append({'domain':domain,'n':len(a),'agreement':sum(x==y for x,y in zip(a,b))/len(a),'kappa':cohen_kappa_score(a,b)})
dump('results/judge_reliability_long.json' if expanded else 'results/judge_reliability.json',metrics);print(metrics)
