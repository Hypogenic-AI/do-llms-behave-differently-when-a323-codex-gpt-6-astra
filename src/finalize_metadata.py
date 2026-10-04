"""Update observed provider and replay metadata from cached real responses."""
import json,collections
from pathlib import Path
from common import dump
from analyze import value
providers=collections.Counter()
for p in Path('results/raw').glob('*.json'):
    r=json.load(open(p));providers[(r['request']['model'],r.get('response',{}).get('provider','error'))]+=1
manifest=json.load(open('results/model_manifest.json'));manifest['observed_providers']=[{'model':m,'provider':p,'records':n} for (m,p),n in sorted(providers.items())];dump('results/model_manifest.json',manifest)
old=json.load(open('results/api_scored_long.json'));rep=json.load(open('results/api_replay_scored.json'));lookup={(r['id'],r['model'],r['style'],r['label']):r for r in old if r['kind']=='behavior'}
summary=json.load(open('results/api_replay_summary.json'));summary['metric_match']=sum(value(r)==value(lookup[(r['id'],r['model'],r['style'],r['label'])]) for r in rep);dump('results/api_replay_summary.json',summary)
print('Finalized provider and replay metadata')
