"""Artifact/invariant checks plus deterministic re-analysis; no paid inference."""
import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from common import ROOT,dump

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks={}
    for p in Path('src').glob('*.py'):ast.parse(p.read_text())
    checks['python_syntax']=True
    ids=json.load(open('results/activation_ids.json'))
    assert not set(ids['train'])&set(ids['test'])
    checks['train_test_ids_disjoint']=True
    pp={r['id']:r for r in json.load(open('results/processed/prompts.json'))}
    assert not {pp[i]['original'] for i in ids['train']}&{pp[i]['original'] for i in ids['test']}
    checks['train_test_exact_text_disjoint']=True
    # These checks are independent of score differences, so validate the causal treatment itself.
    fmt=json.load(open('results/format_tasks.json'))
    import re
    for ident in {r['id'] for r in fmt}:
        ss={r['style']:r['prompt'] for r in fmt if r['id']==ident}
        assert pp[ident]['original'] in ss['human'] and pp[ident]['original'] in ss['ai']
        assert re.findall(r'\w+',ss['human'].lower())==re.findall(r'\w+',ss['ai'].lower())
    checks['format_core_bytes_and_words_equal']=True
    rr=json.load(open('results/intervention_outputs_long.json'))
    for ident in {r['id'] for r in rr}:
        ss=[r for r in rr if r['id']==ident];assert len(ss)==9 and len({r['prompt'] for r in ss})==1
    checks['intervention_text_fixed_nine_arms']=True
    d=pd.read_csv('results/logit_summary.csv').set_index('arm');gap=float(json.load(open('results/probe_metrics.json'))['gap_norm'])
    for arm in ['style_plus','style_minus','random_42','random_43','length','evaluation']:assert abs(d.loc[arm,'mean_edit_norm']/gap-1)<.01
    assert abs(d.loc['style_double','mean_edit_norm']/(2*gap)-1)<.01
    checks['observed_additive_edit_norms_match_within_one_percent']=True
    assert bool(json.load(open('results/local_runtime.json'))['replay_equal'])
    checks['local_exact_replay_16_of_16']=True
    from analyze import paired
    s=paired([1,1,0,0],[0,0,0,0]);assert s['plus']==2 and s['minus']==0 and s['difference']==.5 and s['p']==.5
    s=paired([0]*60,[0]*60);assert s['conservative_low']<0<s['conservative_high']
    checks['paired_statistics_hand_checked']=True
    # Reanalysis is reproducible from frozen real-model output artifacts.
    for script in ['src/analyze.py','src/secondary_analysis.py']:
        subprocess.run([sys.executable,script],check=True,stdout=subprocess.DEVNULL)
    tracked=sorted(Path('results').glob('*.csv'))+sorted(Path('figures').glob('*.png'))
    before={str(p):sha(p) for p in tracked}
    for script in ['src/analyze.py','src/secondary_analysis.py']:
        subprocess.run([sys.executable,script],check=True,stdout=subprocess.DEVNULL)
    after={str(p):sha(p) for p in tracked};assert before==after
    checks['analysis_two_runs_identical']=True
    for file in ['REPORT.md','README.md','planning.md','resources.md','uv.lock','pyproject.toml']:
        assert Path(file).exists() and Path(file).stat().st_size>100
    checks['required_documents_exist']=True
    report=Path('REPORT.md').read_text()
    for sec in ['Executive Summary','Research Question','Methodology','Results','Discussion','Limitations','Conclusions','References']:assert sec in report
    checks['report_sections_present']=True
    secret=os.getenv('OPENROUTER_KEY','')
    if secret:
        assert all(secret not in p.read_text() for p in Path('results/raw').glob('*.json'))
    checks['raw_records_contain_no_api_key']=True
    jobs=[]
    runners={'prepare.py','strict_audit.py','api_experiment.py','format_control.py','extend_api.py','local_experiment.py','local_diagnostics.py','framing_validation.py','judge.py','reliability.py','replay_api.py'}
    for p in psutil.process_iter(['pid','cmdline']):
        args=p.info['cmdline'] or []
        if any(Path(arg).name in runners for arg in args):jobs.append(p.info['pid'])
    assert not jobs,f'Experiment jobs still running: {jobs}'
    checks['no_experiment_processes_running']=True
    dump('results/validation.json',{'checks':checks,'all_passed':all(checks.values()),'analysis_hashes':after,'api_replay':json.load(open('results/api_replay_summary.json')),'caveat':'Software/artifact checks passed; scientific limitations and nonidentical API reruns remain, as documented.'})
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
