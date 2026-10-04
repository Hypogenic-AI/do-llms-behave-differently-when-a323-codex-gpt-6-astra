"""Build manuscript tables and plot from saved results; no model calls or new tests."""
from pathlib import Path
import csv, json, unicodedata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parents[1]
R=P.parent

def read(name):
    return list(csv.DictReader((R/'results'/name).open()))
rows=read('paired_effects.csv')
models=['google/gemini-3.1-flash-lite','qwen/qwen3.5-9b']
domains=['harmful','benign','sycophancy']
short={models[0]:'Gemini',models[1]:'Qwen3.5-9B'}
domain={'harmful':'Harmful','benign':'Benign','sycophancy':'Opinion','calibration':'Calibration'}
def pct(x):return f'{100*float(x):.1f}'
def diff(x):return f'{100*float(x):+.1f}'
def ci(r, prefix='ci'):return '$['+pct(r[prefix+'_low'])+', '+pct(r[prefix+'_high'])+']$'
def select(exp,kind):
    return [next(r for r in rows if r['experiment']==exp and r['subset']=='all' and r['kind']==kind and r['model']==m and r['domain']==d) for m in models for d in domains]
def table(name,caption,cols,header,body,wide=False):
    t='\\begin{table}[t]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{tab:'+name+'}\n'
    if wide:t+='\\resizebox{\\textwidth}{!}{%\n'
    t+='\\begin{tabular}{@{}'+cols+'@{}}\n\\toprule\n'+header+' \\\\\n\\midrule\n'
    t+='\n'.join(' & '.join(r)+r' \\' for r in body)
    t+='\n\\bottomrule\n\\end{tabular}\n'+('}\n' if wide else '')+'\\end{table}\n'
    (P/'tables'/f'{name}.tex').write_text(t)
for exp in ['format','paraphrase']:
    body=[]
    for r in select(exp,'behavior'):
        body.append([short[r['model']],domain[r['domain']],r['n'],pct(r['control_rate']),pct(r['treatment_rate']),diff(r['difference']),ci(r),ci(r,'conservative'),f"{float(r['p_holm']):.3f}"])
    caption=('Exact-content formatting' if exp=='format' else 'Exploratory all-successful-rewrite')+' comparisons. H/A are human-/AI-style endpoint rates (\\%). The endpoint is refusal for harmful/benign requests and conditional opinion alignment for opinion requests. Differences and intervals are in percentage points. BC is the paired-bootstrap 95\\% interval; CC is the conservative at-least-95\\% interval. Holm correction covers six primary tests. Gemini denotes Gemini 3.1 Flash Lite.'
    if exp=='paraphrase':caption+=' Failed semantic controls prevent authorship attribution.'
    table(exp,caption,'llrrrrrrr',r'Model & Domain & $n$ & H & A & $\Delta$ & BC & CC & Holm $p$',body,True)
body=[[domain[r['domain']]]+[r[k] for k in ['sampled','rewritten','first_audit','pair_equivalent','strict']] for r in read('attrition.csv')]
table('attrition','Rewrite attrition by domain. Counts are request pairs; calibration is separate from the benchmark. Only seven benchmark pairs survive the strict gate. Audits are automated, not human gold labels.','lrrrrr','Domain & Sampled & Rewritten & First audit & Pair equivalent & Strict',body)
body=[[short[r['model']],domain[r['domain']],r['n'],diff(r['difference']),ci(r,'conservative'),f"{float(r['p']):.3f}"] for r in read('label_effects.csv')]
table('labels','Exploratory declared-source effects on identical human-style bodies: AI label minus human label. Endpoints follow table~\\ref{tab:format}. The 20 available requests per domain shrink to 13 complete-choice opinion pairs per model. Intervals are conservative at-least-95\\%; $p$ is unadjusted and exact.','llrrrr',r'Model & Domain & $n$ & $\Delta$ (pp) & Interval (pp) & Exact $p$',body)
body=[]
for exp in ['paraphrase','format']:
    for r in select(exp,'perception'):
        body.append([exp.capitalize(),short[r['model']],domain[r['domain']],r['n'],pct(r['control_rate']),pct(r['treatment_rate']),diff(r['difference'])])
table('perception','Separate-session API authorship-style classifications. H/A columns give the percentage classified AI under human-/AI-style prompts. These are elicited binary outputs, not calibrated probabilities or verified beliefs during responding. Paired counts reflect available valid judgments.','lllrrrr',r'Arm & Model & Domain & $n$ & H called AI (\%) & A called AI (\%) & $\Delta$ (pp)',body,True)
metrics=json.loads((R/'results/probe_metrics.json').read_text())
body=[['Logistic activation probe',r'\textbf{0.990}','AUPRC .989'],['Mean-difference direction','.970','95\\% CI [.954, .984]'],['Surface-only logistic classifier','.970','Seven measured features'],['Shuffled-label activation probe','.495','Negative control'],['Surface-residual activation probe','.665','Training-only nuisance fit'],['Strict-calibration mean direction','.965','18 calibration pairs'],['Unseen-writer logistic transfer','.970 / .972','Also changes domain mixture']]
table('probes','Held-out local Qwen3-8B style discrimination on 124 benchmark pairs. The standard calibration split has 77 first-audit-accepted pairs. Bold marks the highest reported AUROC, not evidence of a causal mechanism. Only the mean-direction row has a reported request-bootstrap interval.','lrl','Readout & AUROC & Diagnostic',body)
iv=read('intervention_effects.csv')
arms=['sham','style_plus','style_minus','style_double','style_ablate','random_42','random_43','length','evaluation']
names=['Sham','Style $+g$','Style $-g$','Style $+2g$','Style centering','Random draw 1','Random draw 2','Length','Evaluation framing']
body=[]
for arm,name in zip(arms,names):
    rr=[next(r for r in iv if r['arm']==arm and r['domain']==d) for d in domains]
    body.append([name]+[diff(r['difference']) for r in rr]+[rr[-1]['n']])
table('interventions','Local fixed-text intervention effects relative to sham (percentage points). Harmful/benign columns measure refusal on 20 paired requests each. Opinion measures conditional alignment with paired count $n_O$. All exact endpoint $p$-values equal 1; every harmful interval is approximately $[-19.7,19.7]$ points. Single-dose additive arms match norm $g$; double style and sample-dependent centering have different norms.','lrrrr',r'Intervention & Harmful & Benign & Opinion & $n_O$',body)

# Vector plot with conservative behavioral intervals, directly from saved estimates.
plt.rcParams.update({'font.size':11.5,'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.3,3.4),gridspec_kw={'width_ratios':[1,1.2]})
colors=['#2574a9']*3+['#bd5a26']*3
p=select('format','perception'); b=select('format','behavior')
labels=[short[r['model']]+' / '+domain[r['domain']] for r in p]
for ax in axs:
    ax.axvline(0,color='.45',lw=.8)
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(5.6,-.7)
    ax.set_yticks(range(6))
    ax.set_xlabel('AI-style minus human-style (pp)')
for i,(r,s,c) in enumerate(zip(p,b,colors)):
    x=100*float(r['difference']); axs[0].plot(x,i,'o',color=c)
    axs[0].text(x+3,i,f"{x:+.1f} (n={r['n']})",va='center',fontsize=9)
    x=100*float(s['difference']); lo=100*float(s['conservative_low']); hi=100*float(s['conservative_high'])
    axs[1].errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color=c,capsize=3)
    axs[1].text(29,i,f"n={s['n']}",va='center',fontsize=9)
axs[0].set_yticklabels(labels);axs[1].set_yticklabels([])
axs[0].set_xlim(-6,132);axs[0].set_xticks([0,40,80,120])
axs[1].set_xlim(-22,38);axs[1].set_xticks([-20,-10,0,10,20,30])
for x in [-10,10]:axs[1].axvline(x,color='.8',ls='--',lw=.8,zorder=0)
axs[0].set_title('AI-authorship classification',fontweight='bold',fontsize=10)
axs[1].set_title('Behavior (conservative intervals)',fontweight='bold',fontsize=10)
fig.tight_layout(w_pad=1.3)
fig.savefig(P/'figures/format_dissociation.pdf',bbox_inches='tight')
fig.savefig(P/'figures/format_dissociation.png',dpi=180,bbox_inches='tight')

# Bibliography from the supplied catalog; no inferred publication venues.
keys={'2310.13548':'sharma2023sycophancy','2212.09251':'perez2022evaluations','2603.19426':'devbunova2026format','2505.23836':'needham2025evaluated','2505.14617':'abdelnabi2025hawthorne','2507.01786':'nguyen2025probing','2608.21766':'heidari2026awareness','2608.24780':'quaremba2026probing','2508.13152':'chen2025repreguard','2404.13076':'panickssery2024recognize','2509.00462':'xu2025hiring','2407.12856':'laurito2024bias','2608.11008':'chalkidis2026templated'}
def latex(s):
    accents={'\u0301':"'",'\u0300':'`','\u0302':'^','\u0303':'~','\u0308':'"','\u030c':'v','\u0304':'=','\u0307':'.','\u0328':'k','\u0327':'c'}
    s=s.replace('&',r'\&').replace('–','--').replace('—','---')
    out=''
    for ch in s:
        if ord(ch)<128:out+=ch;continue
        n=unicodedata.normalize('NFD',ch)
        if len(n)==2 and n[1] in accents:out+='{\\'+accents[n[1]]+' '+n[0]+'}'
        else:out+=ch
    return out
entries=[]
for rec in json.loads((R/'notes/papers_catalog.json').read_text()):
    if rec['id'] not in keys:continue
    title=rec['title'][0] if isinstance(rec['title'],list) else rec['title']
    date=rec['date'][0] if isinstance(rec['date'],list) else rec['date']
    entries.append('@misc{'+keys[rec['id']]+',\n  title = {{'+latex(title)+'}},\n  author = {'+latex(' and '.join(rec['author']))+'},\n  year = {'+date[:4]+'},\n  howpublished = {arXiv:'+rec['id']+'},\n  eprint = {'+rec['id']+'},\n  archiveprefix = {arXiv},\n  url = {'+rec['url']+'}\n}\n')
for key,author,title,url in [
('jbbdata','JailbreakBench','JBB-Behaviors','https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors'),
('anthropicdata','Anthropic','Original sycophancy evaluation files','https://github.com/anthropics/evals/tree/84fcc677e52e1902d696c32cd1a6b663e70d3993/sycophancy'),
('wildchatdata','Allen Institute for AI','WildChat-1M','https://huggingface.co/datasets/allenai/WildChat-1M'),
('qwenmodel','Qwen Team','Qwen3-8B model card','https://huggingface.co/Qwen/Qwen3-8B'),
('openrouterdocs','OpenRouter','API reference overview','https://openrouter.ai/docs/api_reference/overview')]:
    entries.append('@misc{'+key+',\n  author = {{'+author+'}},\n  title = {{'+title+'}},\n  howpublished = {Online resource, study snapshot},\n  year = {2026},\n  note = {Study snapshot accessed October 4, 2026},\n  url = {'+url+'}\n}\n')
(P/'references.bib').write_text('\n'.join(sorted(entries)))
print('Generated seven tables, vector plot, and 18 references from supplied artifacts.')
