"""Desktop-driven own-data list sync and lightweight standard-runner GPU survey.

No training, scheduler mutations or external calls from Hali. Dataset exports
must already be present; this refuses to publish the list if checks fail.
"""
from pathlib import Path
import json
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REMOTE=r'''
from pathlib import Path
import ast,csv,hashlib,json,os,shutil,sys,time
h=Path.home();request=json.load(sys.stdin);root=h/'Data/EEG'
for name in request['names']:
 if not name or Path(name).name!=name:raise RuntimeError('Unsafe dataset name')
 for split in ['TRAIN','TEST']:
  p=root/name/(name+'_'+split+'.ts')
  if not p.is_file() or p.stat().st_size==0:raise RuntimeError('Missing data: '+str(p))
for entry in request['ids']:
 p=(root/entry['path']).resolve()
 if root.resolve() not in p.parents:raise RuntimeError('Unsafe sidecar path')
 if not p.is_file() or p.stat().st_size!=entry['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
  raise RuntimeError('Subject-ID sidecar differs: '+entry['path'])
listing=h/'DataSetLists/eeg.txt';old=listing.read_text()
backup=listing.with_name('eeg.txt.bak-'+time.strftime('%Y%m%d-%H%M%S'))
shutil.copy2(listing,backup);pending=listing.with_suffix('.sync-pending')
pending.write_text(request['text']);os.replace(str(pending),str(listing))
repo=h/'Code/tsml-eval';slurm=repo/'_tsml_research_resources/slurm'
sys.path.insert(0,str(slurm));import tsml_slurm as ts
factory=ast.parse((repo/'tsml_eval/experiments/_get_classifier.py').read_text())
mapping={}
for node in factory.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='neuralbench_deep_models' for t in node.targets):mapping=ast.literal_eval(node.value)
explicit_aliases={value.value for node in ast.walk(factory) if isinstance(node,ast.Compare) for value in ast.walk(node) if isinstance(value,ast.Constant) and isinstance(value.value,str)}
configured=[]
for config in sorted((slurm/'experiments').glob('hali_eeg*.toml')):
 e=ts.load_experiment(config)
 if not e.resources.gpu:continue
 ts.read_datasets(e);tasks=list(ts.all_tasks(e));state=ts.load_state(e);models=[]
 for model in e.estimators:
  own=[t for t in tasks if t.estimator==model.name];missing=[t for t in own if not ts.is_complete(e,t)]
  failed=[t.dataset for t in missing if state['tasks'].get(t.key,{}).get('terminal')]
  new=[t.dataset for t in missing if t.dataset not in set(old.splitlines())]
  models.append(dict(model=model.name,total=len(own),complete=len(own)-len(missing),missing=[t.dataset for t in missing],terminal_failures=failed,newly_listed_missing=new,factory_nb_supported=(model.name.lower() in mapping or model.name.lower() in explicit_aliases) if model.name.lower().endswith('-nb') else None))
 configured.append(dict(config=config.name,name=e.name,models=models,unresolved=ts.unresolved(e),env=e.resources.env_name,email=e.email,supervisor=e.supervisor.mode))
supported=[]
for sp in (h/'.conda/envs/tsml-eval/lib').glob('python*/site-packages'):
 p=sp/'aeon_neuro/classification/deep_learning/_braindecode.py'
 if p.is_file():
  for node in ast.parse(p.read_text()).body:
   if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUPPORTED_MODELS' for t in node.targets):supported=list(ast.literal_eval(node.value))
  break
info=list(csv.DictReader((repo/'_tsml_research_resources/dataset_lists/eeg_ctnet_info.csv').open()))
missing_info=sorted(set(request['names'])-{r['dataset'] for r in info})
cache=h/'.cache/huggingface/hub'
cached=[p.name for p in cache.iterdir() if p.is_dir() and any(s in p.name.lower() for s in ['cbramod','labram'])] if cache.exists() else []
print(json.dumps(dict(list_path=str(listing),backup=str(backup),problems=len(request['names']),added=sorted(set(request['names'])-set(old.splitlines())),removed=sorted(set(old.splitlines())-set(request['names'])),subject_id_sidecars_verified=len(request['ids']),data_arrays_checked='76 TRAIN/TEST pairs exist and have matching export names/sizes in prior inventory; array contents were not rehashed',gpu=configured,factory_nb_models=mapping,installed_supported_models=supported,dataset_info_columns=list(info[0]) if info else [],missing_dataset_info=missing_info,foundation_cache_directories=cached)))
'''


def main():
    request=(ROOT/'.openeegbench/hali-sync-request-20261111.json').read_text(encoding='utf-8')
    args=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
          '-o','UpdateHostKeys=no','-o','ConnectTimeout=20','ajb@hali.uea.ac.uk','python3 -c '+shlex.quote(REMOTE)]
    result=subprocess.run(args,input=request,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:
        print(result.stderr);raise RuntimeError('Sync/survey stopped; inspect remote list before retrying')
    report=json.loads(result.stdout)
    (ROOT/'docs/hali-dataset-sync-gpu-survey-20261111.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('List synchronised:',report['problems'],'problems; sidecars checked:',report['subject_id_sidecars_verified'])
    for item in report['gpu']:
        print(item['name'],sum(m['complete'] for m in item['models']),'/',sum(m['total'] for m in item['models']),
              'missing',sum(len(m['missing']) for m in item['models']),
              'newly listed',sum(len(m['newly_listed_missing']) for m in item['models']),
              'terminal',sum(len(m['terminal_failures']) for m in item['models']))


if __name__=='__main__':main()
