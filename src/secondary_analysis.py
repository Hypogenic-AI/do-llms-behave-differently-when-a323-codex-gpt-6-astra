"""Robustness checks and diagnostics; all exploratory, not extra confirmatory tests."""
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from analyze import paired,value,PROMPTS,holm
from common import dump

def main():
    out=[];budget=[];examples=[];fulfillment=[]
    for name in ['api','format','local']:
        old=json.load(open(f'results/{name}_scored.json'));new=json.load(open(f'results/{name}_scored_long.json'))
        for version,rows in [('initial',old),('extended',new)]:
            for model in sorted({r['model'] for r in rows}):
                for domain in ['harmful','benign','sycophancy']:
                    rr=[r for r in rows if r['model']==model and r['domain']==domain and r['kind']=='behavior' and r['label']=='none']
                    budget.append({'experiment':name,'version':version,'model':model,'domain':domain,'n':len(rr),'truncated':sum(r.get('finish_reason')=='length' for r in rr),'invalid':sum(value(r) is None for r in rr),'provider_errors':sum(bool(r.get('error')) for r in rr)})
        for model in sorted({r['model'] for r in new}):
            for domain in ['harmful','benign','sycophancy']:
                rr=[r for r in new if r['model']==model and r['domain']==domain and r['kind']=='behavior' and r['label']=='none' and r['style'] in ['ai','human']]
                by={}
                for r in rr:by.setdefault(r['id'],{})[r['style']]=r
                if domain!='sycophancy':
                    good=[p for _,p in sorted(by.items()) if len(p)==2 and all(value(r,'fulfillment') is not None for r in p.values())]
                    fulfillment.append({'experiment':name,'model':model,'domain':domain,**paired([value(p['ai'],'fulfillment') for p in good],[value(p['human'],'fulfillment') for p in good])})
                for variant in ['invalid_as_nonmatching','complete_only','tight_surface']:
                    a=[];b=[]
                    for ident,p in sorted(by.items()):
                        if set(p)!= {'ai','human'}:continue
                        if variant=='complete_only' and any(r.get('finish_reason')!='stop' for r in p.values()):continue
                        if variant=='tight_surface':
                            if name=='format':continue
                            r=PROMPTS[ident];audit=r.get('audit',{})
                            if not (.9<=r.get('length_ratio',0)<=1.1 and audit.get('a_politeness')==audit.get('b_politeness') and audit.get('a_clarity')==audit.get('b_clarity')):continue
                        av=value(p['ai']);bv=value(p['human'])
                        if variant=='invalid_as_nonmatching' and domain=='sycophancy':av=av or 0;bv=bv or 0
                        if av is None or bv is None:continue
                        a.append(av);b.append(bv)
                    out.append({'experiment':name,'model':model,'domain':domain,'sensitivity':variant,**paired(a,b)})
                # Preserve benign examples only in compact error report; raw files contain full data.
                if domain=='benign':
                    for ident,p in sorted(by.items()):
                        if len(p)==2 and value(p['ai'])!=value(p['human']):
                            examples.append({'experiment':name,'model':model,'id':ident,'request':PROMPTS[ident]['original'],'human_response':p['human']['text'],'ai_response':p['ai']['text'],'human_score':p['human'].get('score'),'ai_score':p['ai'].get('score')})
    for family in ['api','format','local']:
        holm([r for r in fulfillment if r['experiment']==family])
        holm([r for r in out if r['experiment']==family and r['sensitivity']=='invalid_as_nonmatching' and r['domain']=='sycophancy'])
    pd.DataFrame(fulfillment).to_csv('results/fulfillment_effects.csv',index=False)
    pd.DataFrame(out).to_csv('results/sensitivity.csv',index=False);pd.DataFrame(budget).to_csv('results/budget_diagnostics.csv',index=False);dump('results/benign_discordant_examples.json',examples)
    if Path('results/logit_diagnostics.json').exists():
        d=pd.DataFrame(json.load(open('results/logit_diagnostics.json')))
        d.groupby('arm').agg(mean_kl=('kl_nats','mean'),median_kl=('kl_nats','median'),max_kl=('kl_nats','max'),top_token_change=('top_token_changed','mean'),mean_edit_norm=('actual_delta_norm','mean')).to_csv('results/logit_summary.csv')
    if Path('results/intervention_perception.json').exists():
        rows=json.load(open('results/intervention_perception.json'));dd=[]
        for domain in ['harmful','benign','sycophancy']:
            rr=[r for r in rows if r['domain']==domain];base={r['id']:r for r in rr if r['intervention']=='sham'}
            for arm in sorted({r['intervention'] for r in rr}):
                ss=[r for r in rr if r['intervention']==arm];valid=[r for r in ss if value(r) is not None and value(base[r['id']]) is not None]
                dd.append({'domain':domain,'arm':arm,**paired([value(r) for r in valid],[value(base[r['id']]) for r in valid])})
        pd.DataFrame(dd).to_csv('results/intervention_perception_effects.csv',index=False)
    from sklearn.metrics import roc_auc_score
    a=np.load('results/activations.npz');ids=json.load(open('results/activation_ids.json'))
    ii=np.repeat([PROMPTS[i].get('strict_accepted',False) for i in ids['train']],2);x=a['train'][ii];y=a['train_labels'][ii]
    v=x[y==1].mean(0)-x[y==0].mean(0);v/=np.linalg.norm(v)
    rob=json.load(open('results/probe_robustness.json'));rob['strict_train_n']=int(ii.sum()/2);rob['strict_train_auc']=float(roc_auc_score(a['test_labels'],a['test']@v))
    vs=a['style'];ev=a['evaluation'];n=a['length'];vr=vs-(vs@n)*n;er=ev-(ev@n)*n
    rob['style_eval_cos_after_length_projection']=float(vr@er/(np.linalg.norm(vr)*np.linalg.norm(er)));dump('results/probe_robustness.json',rob)
    costs=[]
    for name in ['api','format']:
        for r in json.load(open(f'results/{name}_outputs_long.json')):
            raw=json.load(open('results/raw/'+r['key']+'.json'));u=raw.get('response',{}).get('usage',{})
            costs.append({'experiment':name,'model':r['model'],'kind':r['kind'],'style':r['style'],'label':r['label'],'cost_usd':u.get('cost',0) or 0,'prompt_tokens':u.get('prompt_tokens',0),'completion_tokens':u.get('completion_tokens',0)})
    pd.DataFrame(costs).groupby(['experiment','model','kind','style','label']).sum().to_csv('results/condition_costs.csv')
    surface=[]
    for r in PROMPTS.values():
        if 'human' not in r:continue
        au=r.get('audit',{});swap=r.get('audit_swap',0)
        surface.append({'id':r['id'],'domain':r['domain'],'writer':r['writer'],'original_words':len(r['original'].split()),'human_words':len(r['human'].split()),'ai_words':len(r['ai'].split()),'ratio_ai_human':r['length_ratio'],'clarity_difference':(au['a_clarity']-au['b_clarity']) if swap else (au['b_clarity']-au['a_clarity']),'politeness_difference':(au['a_politeness']-au['b_politeness']) if swap else (au['b_politeness']-au['a_politeness'])})
    pd.DataFrame(surface).to_csv('results/surface_controls.csv',index=False)
    print('Secondary analysis complete; benign discordant examples',len(examples))
if __name__=='__main__':main()
