"""Record live model availability and isolated runtime before a new acquisition."""
import json,sys,platform,importlib.metadata,subprocess
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
assert Path.cwd()==ROOT,'Run from workspace root'
for d in ['results/raw','results/processed','figures','logs']:Path(d).mkdir(parents=True,exist_ok=True)
from common import dump,GEM,QWEN,WRITER
r=requests.get('https://openrouter.ai/api/v1/models',timeout=60);r.raise_for_status();cat=r.json()
assert {GEM,QWEN,WRITER}<={r['id'] for r in cat['data']},'Planned model unavailable: revise protocol before proceeding'
dump('results/live_catalog.json',cat)
# Historical mechanistic baseline is intentionally pinned, not silently upgraded.
dump('results/local_model_config.json',{'model':'Qwen/Qwen3-8B','revision':'b968826d9c46dd6066d109eabc6255188de91218'})
import datetime
from datetime import timezone
dump('results/model_manifest.json',{'catalog_access_date':datetime.datetime.now(timezone.utc).date().isoformat(),'routing':'OpenRouter automatic routing; no provider pinned','models':[r for r in cat['data'] if r['id'] in [GEM,QWEN,WRITER]],'local':json.load(open('results/local_model_config.json'))})
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv'],capture_output=True,text=True)
dump('results/environment.json',{'python':sys.version,'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','pandas','scikit-learn','scipy','matplotlib','requests','huggingface-hub']},'gpu':gpu.stdout,'virtualenv':'.venv','seed':42,'torch_workaround':'Disable eager Triton overrides if available; use standard ATen CUDA kernels.'})
print('Verified live catalog and recorded environment')
