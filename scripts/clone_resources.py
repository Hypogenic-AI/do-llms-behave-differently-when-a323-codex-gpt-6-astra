import pathlib,subprocess,json
R=pathlib.Path.cwd();assert (R/'.git').is_dir();print(R,flush=True)
repos={'jailbreakbench':'JailbreakBench/jailbreakbench','mgt_probes':'gerritq/mgt_probes','eval-awareness-format':'viliana-dev/eval-awareness-format','test-awareness-steering':'microsoft/Test_Awareness_Steering','ai-ai-bias':'lauritowal/ai-ai-bias','evaluation-awareness':'evaluation-awareness/evaluation-awareness'}
out=[]
for name,repo in repos.items():
 p=R/'code'/name;u='https://github.com/'+repo+'.git'
 try:
  if not p.exists():subprocess.run(['git','clone','--depth','1',u,str(p)],check=True,capture_output=True,timeout=150)
  commit=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip();out.append(dict(name=name,url=u,path='code/'+name,commit=commit));print(name,commit,flush=True)
 except Exception as e:out.append(dict(name=name,url=u,error=str(e)));print(name,str(e),flush=True)
pathlib.Path('notes/code_manifest.json').write_text(json.dumps(out,indent=2))
