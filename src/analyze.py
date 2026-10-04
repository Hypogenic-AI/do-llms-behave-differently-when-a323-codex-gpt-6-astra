"""Request-paired inference, attrition, diagnostics, and publication-ready figures."""
import json,re,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binomtest,beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import dump,SEED
RNG=np.random.default_rng(SEED)
PROMPTS={r['id']:r for r in json.load(open('results/processed/prompts.json'))}

def paired(a,b):
    """AI/treatment minus human/control, with paired bootstrap and exact discordance test.
    Conservative simultaneous binomial bounds also cover zero-discordance samples.
    """
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);d=a-b;n=len(d)
    if not n:return {'n':0}
    boots=d[RNG.integers(0,n,size=(10000,n))].mean(1)
    plus=int(((a==1)&(b==0)).sum());minus=int(((a==0)&(b==1)).sum())
    def bounds(k):return (0 if k==0 else float(beta.ppf(.0125,k,n-k+1)),1 if k==n else float(beta.ppf(.9875,k+1,n-k)))
    lp,up=bounds(plus);lm,um=bounds(minus)
    return {'n':n,'treatment_rate':a.mean(),'control_rate':b.mean(),'difference':d.mean(),'ci_low':float(np.quantile(boots,.025)),'ci_high':float(np.quantile(boots,.975)),
        'conservative_low':lp-um,'conservative_high':up-lm,'plus':plus,'minus':minus,'p':float(binomtest(plus,plus+minus,.5).pvalue) if plus+minus else 1.,'zero_discordance':plus+minus==0}

def holm(rows):
    order=np.argsort([r.get('p',1) for r in rows]);last=0
    for rank,i in enumerate(order):last=max(last,min(1,(len(rows)-rank)*rows[i].get('p',1)));rows[i]['p_holm']=last

def value(r,metric=None):
    if r.get('error'):return None
    s=r.get('score',{})
    if metric=='fulfillment':return int(s['fulfillment']==2) if 'fulfillment' in s else None
    if metric=='invalid':return int(s.get('alignment') is None) if r['domain']=='sycophancy' else int('refusal' not in s)
    if r['kind']=='perception':
        t=r['text'].strip().upper().strip('.');return 1 if t=='AI' else 0 if t=='HUMAN' else None
    return s.get('alignment') if r['domain']=='sycophancy' else s.get('refusal')

def comparisons(rows,experiment,subsets):
    out=[]
    for subset in subsets:
        for model in sorted({r['model'] for r in rows}):
            for domain in ['harmful','benign','sycophancy']:
                for kind in ['behavior','perception']:
                    selected=[r for r in rows if r['model']==model and r['domain']==domain and r['kind']==kind and r['label']=='none' and r['style'] in ['ai','human']]
                    if subset!='all':selected=[r for r in selected if PROMPTS[r['id']].get(subset,False)]
                    by={}
                    for r in selected:by.setdefault(r['id'],{})[r['style']]=value(r)
                    valid=[by[k] for k in sorted(by) if by[k].get('ai') is not None and by[k].get('human') is not None]
                    stat=paired([r['ai'] for r in valid],[r['human'] for r in valid])
                    out.append({'experiment':experiment,'subset':subset,'model':model,'domain':domain,'kind':kind,'n_available':len(by),'n_invalid_pairs':len(by)-len(valid),**stat})
    for subset in subsets:
        for kind in ['behavior','perception']:holm([r for r in out if r['subset']==subset and r['kind']==kind])
    return out

def main():
    apis=json.load(open('results/api_scored_long.json' if Path('results/api_scored_long.json').exists() else 'results/api_scored.json'))
    allstats=comparisons(apis,'paraphrase',['all','accepted','pair_accepted','strict_accepted'])
    fmt=json.load(open('results/format_scored_long.json' if Path('results/format_scored_long.json').exists() else 'results/format_scored.json')) if Path('results/format_scored.json').exists() else []
    if fmt:allstats+=comparisons(fmt,'format',['all'])
    local=[]
    if Path('results/local_scored.json').exists():
        local=json.load(open('results/local_scored_long.json' if Path('results/local_scored_long.json').exists() else 'results/local_scored.json'))+json.load(open('results/local_perception.json'))
        allstats+=comparisons(local,'local_paraphrase',['all','accepted','pair_accepted','strict_accepted'])
    df=pd.DataFrame(allstats);df.to_csv('results/paired_effects.csv',index=False)
    # Secondary source-label effects compare identical bodies.
    labels=[]
    for model in sorted({r['model'] for r in apis}):
        for domain in ['harmful','benign','sycophancy']:
            rr=[r for r in apis if r['model']==model and r['domain']==domain and r['kind']=='behavior' and r['style']=='human' and r['label']!='none'];by={}
            for r in rr:by.setdefault(r['id'],{})[r['label']]=value(r)
            good=[by[k] for k in sorted(by) if by[k].get('human') is not None and by[k].get('AI') is not None]
            labels.append({'model':model,'domain':domain,**paired([r['AI'] for r in good],[r['human'] for r in good])})
    pd.DataFrame(labels).to_csv('results/label_effects.csv',index=False)
    # All arm descriptive statistics retain invalids and truncations.
    desc=[]
    for name,rr in [('paraphrase',apis),('format',fmt),('local',local)]:
        for model in sorted({r['model'] for r in rr}):
            for domain in ['harmful','benign','sycophancy']:
                for style in ['original','human','ai']:
                    ss=[r for r in rr if r['model']==model and r['domain']==domain and r['kind']=='behavior' and r['label']=='none' and r['style']==style]
                    if not ss:continue
                    vv=[value(r) for r in ss];valid=[v for v in vv if v is not None]
                    desc.append({'experiment':name,'model':model,'domain':domain,'style':style,'n':len(ss),'valid':len(valid),'mean':np.mean(valid) if valid else None,'std':np.std(valid,ddof=1) if len(valid)>1 else None,'min':min(valid) if valid else None,'max':max(valid) if valid else None,'truncated':sum(r.get('finish_reason')=='length' for r in ss),'substantial_fulfillment':np.mean([r['score']['fulfillment']==2 for r in ss if 'fulfillment' in r.get('score',{})]) if domain!='sycophancy' else None})
    pd.DataFrame(desc).to_csv('results/descriptives.csv',index=False)
    # Intention-to-manipulate attrition, not silent listwise deletion.
    attr=[]
    for domain in ['harmful','benign','sycophancy','calibration']:
        rr=[r for r in PROMPTS.values() if r['domain']==domain]
        attr.append({'domain':domain,'sampled':len(rr),'rewritten':sum('human'in r for r in rr),'first_audit':sum(r.get('accepted',False) for r in rr),'pair_equivalent':sum(r.get('pair_accepted',False) for r in rr),'strict':sum(r.get('strict_accepted',False) for r in rr)})
    pd.DataFrame(attr).to_csv('results/attrition.csv',index=False)
    # Paired intervention effects relative to sham on fixed text.
    if Path('results/intervention_scored.json').exists():
        rr=json.load(open('results/intervention_scored_long.json' if Path('results/intervention_scored_long.json').exists() else 'results/intervention_scored.json'));ss=[]
        for domain in ['harmful','benign','sycophancy']:
            sub=[r for r in rr if r['domain']==domain];base={r['id']:r for r in sub if r['intervention']=='sham'}
            for arm in sorted({r['intervention'] for r in sub}):
                t=sorted([r for r in sub if r['intervention']==arm],key=lambda r:r['id']);good=[r for r in t if value(r) is not None and value(base[r['id']]) is not None]
                ss.append({'domain':domain,'arm':arm,'n_invalid':len(t)-len(good),**paired([value(r) for r in good],[value(base[r['id']]) for r in good]),'incoherent':sum(r.get('score',{}).get('incoherent',False) for r in t),'truncated':sum(r.get('finish_reason')=='length' for r in t)})
        pd.DataFrame(ss).to_csv('results/intervention_effects.csv',index=False)
        allraw=json.load(open('results/intervention_outputs.json'));mm=[]
        for arm in sorted({r['intervention'] for r in allraw}):
            rrs=[r for r in allraw if r['intervention']==arm and r['domain']=='arithmetic']
            correct=[bool(re.fullmatch(r'\s*'+str(r['answer'])+r'[.!]?\s*',r['text'])) for r in rrs]
            mm.append({'arm':arm,'n':len(rrs),'correct':sum(correct),'accuracy':np.mean(correct)})
        pd.DataFrame(mm).to_csv('results/arithmetic.csv',index=False)
    # Account once for every raw completion, including writers and judges.
    usage=[]
    for p in Path('results/raw').glob('*.json'):
        r=json.load(open(p));u=r.get('response',{}).get('usage',{})
        usage.append({'tag':r['tag'],'model':r['request']['model'],'provider':r.get('response',{}).get('provider'),'cost':u.get('cost',0),'prompt_tokens':u.get('prompt_tokens',0),'completion_tokens':u.get('completion_tokens',0),'error':int('error'in r)})
    pd.DataFrame(usage).groupby(['tag','model']).agg(calls=('cost','size'),cost=('cost','sum'),prompt_tokens=('prompt_tokens','sum'),completion_tokens=('completion_tokens','sum'),errors=('error','sum')).to_csv('results/costs.csv')
    dump('results/usage_total.json',{'calls':len(usage),'cost_usd':sum(x['cost'] or 0 for x in usage),'prompt_tokens':sum(x['prompt_tokens'] for x in usage),'completion_tokens':sum(x['completion_tokens'] for x in usage),'errors':sum(x['error'] for x in usage)})
    # Effect plots use bootstrap intervals; conservative bounds are also in CSV/report.
    f,axs=plt.subplots(1,2,figsize=(12,5),sharex=True)
    for ax,name in zip(axs,['paraphrase','format']):
        dat=df[(df.experiment==name)&(df.subset=='all')&(df.kind=='behavior')].reset_index(drop=True)
        for i,r in dat.iterrows():
            ax.errorbar(100*r['difference'],i,xerr=[[100*(r['difference']-r.ci_low)],[100*(r.ci_high-r['difference'])]],fmt='o',color='tab:blue')
        ax.set_yticks(range(len(dat)),[r.model.split('/')[-1]+' / '+r.domain for _,r in dat.iterrows()]);ax.axvline(0,color='gray',ls='--');ax.set_title(name);ax.set_xlabel('AI-style minus human-style (percentage points)')
    f.suptitle('Refusal (harmful/benign) and opinion alignment (sycophancy)\nPaired 95% bootstrap CIs; zero-discordance intervals are degenerate')
    f.tight_layout();f.savefig('figures/behavior_effects.png',dpi=180);plt.close(f)
    f,ax=plt.subplots(figsize=(8,4));dat=df[(df.subset=='all')&(df.kind=='perception')]
    labels=[r.experiment+' / '+r.model.split('/')[-1]+' / '+r.domain for _,r in dat.iterrows()]
    ax.barh(range(len(dat)),100*dat['difference']);ax.set_yticks(range(len(dat)),labels,fontsize=7);ax.set_xlabel('Change in AI-style classifications (percentage points)');ax.set_title('Separate-session manipulation checks');f.tight_layout();f.savefig('figures/perception.png',dpi=180);plt.close(f)
    if Path('results/activations.npz').exists():
        a=np.load('results/activations.npz');scores=a['test']@a['style'];evalscores=a['test']@a['evaluation']
        f,axs=plt.subplots(1,2,figsize=(10,4));
        for lab,c in [(0,'tab:orange'),(1,'tab:blue')]:axs[0].hist(scores[a['test_labels']==lab],bins=20,alpha=.6,label=['Human style','AI style'][lab],color=c)
        axs[0].legend();axs[0].set_xlabel('AI-style projection (activation units)');axs[0].set_ylabel('Prompt count');axs[1].scatter(scores,evalscores,c=a['test_labels'],s=12);axs[1].set_xlabel('AI-style projection');axs[1].set_ylabel('Evaluation-framing projection');f.tight_layout();f.savefig('figures/representations.png',dpi=180);plt.close(f)
    print(df[(df.subset=='all')&(df.kind=='behavior')].to_string(index=False));print('usage',json.load(open('results/usage_total.json')))
if __name__=='__main__':main()
