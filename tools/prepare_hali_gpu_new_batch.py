"""Prepare the standard-controller new-data GPU batch on the desktop."""
from pathlib import Path
import json
import shutil
import tarfile
import tomllib

ROOT=Path(__file__).resolve().parents[1]


def table(name,values):
    return '['+name+']\n'+''.join(k+' = '+json.dumps(v)+'\n' for k,v in values.items())+'\n'


def prepare():
    root=ROOT/'.openeegbench/hali-new-gpu-20261111';root.mkdir(parents=True,exist_ok=True)
    evidence=json.loads((ROOT/'.openeegbench/hali-sync-inventory-20261111.json').read_bytes())
    models=[];kwargs={}
    sources=['hali_eeg_ctnet.toml','hali_eeg_eegnet_nb.toml','hali_eeg_nb_deep.toml','hali_eeg_nb_more.toml','hali_eeg_nb_more2.toml']
    for name in sources:
        config=tomllib.loads(evidence['configs'][name])
        models.extend(config['estimators']['DeepLearning']);kwargs.update(config['estimator_kwargs'])
    assert len(models)==len(set(models))==23
    remote='/gpfs/home/ajb/tsml-gpu-new-20261111'
    exp=dict(name='eeg-new-gpu-20261111',cluster='hali',resources='gpu',task='classification',
        repo_dir='/gpfs/home/ajb/Code/tsml-eval',branch='ajb/slurm-runner',pin_commit=True,
        data_dir='/gpfs/home/ajb/Data/EEG',datasets=remote+'/datasets.txt',
        dataset_info='/gpfs/home/ajb/Code/tsml-eval/_tsml_research_resources/dataset_lists/eeg_ctnet_info.csv',
        results_dir='/gpfs/home/ajb/Results/EEG',state_dir=remote+'/state',resamples=1,
        smallest_datasets_first=True,priority='dataset',email='')
    setup='export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled WANDB_DISABLED=true DO_NOT_TRACK=1'
    resources=dict(env_name='tsml-eval',gpu_check='torch',memory_gb=[64,128],max_active=16,setup=[setup])
    text=table('experiment',exp)+table('estimators',dict(DeepLearning=models))
    for model,params in kwargs.items():text+=table('estimator_kwargs.'+json.dumps(model),params)
    text+=table('slurm',resources)
    (root/'experiment.toml').write_text(text,encoding='utf-8',newline='\n')
    (root/'datasets.txt').write_text('ChildhoodSeizures\nClinicalSleepStages\nFlickerSymbols\nObsessiveCompulsiveSymptoms\nSymbolAttention\n',encoding='utf-8')
    slurm=Path('C:/Code/tsml-eval/_tsml_research_resources/slurm');(root/'slurm/clusters').mkdir(parents=True,exist_ok=True)
    controller=(slurm/'tsml_slurm.py').read_text(encoding='utf-8')
    old='        "#SBATCH --mail-type=FAIL",'
    assert controller.count(old)==1
    controller=controller.replace(old,'        "#SBATCH --mail-type=FAIL" if e.email else "#SBATCH --mail-type=NONE",')
    (root/'slurm/tsml_slurm.py').write_text(controller,encoding='utf-8',newline='\n')
    shutil.copy2(slurm/'clusters/hali.toml',root/'slurm/clusters/hali.toml')
    shutil.copy2(ROOT/'tools/check_hali_gpu_shapes.py',root/'shape_preflight.py')
    script='''#!/bin/bash
#SBATCH --job-name=eeg-new-gpu-preflight
#SBATCH --account=cmp
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/gpfs/home/ajb/tsml-gpu-new-20261111/preflight-%j.out
#SBATCH --error=/gpfs/home/ajb/tsml-gpu-new-20261111/preflight-%j.err
set -euo pipefail
ROOT=/gpfs/home/ajb/tsml-gpu-new-20261111
PY=/gpfs/home/ajb/.conda/envs/tsml-eval/bin/python
export PATH=/opt/slurm/25.11.4/bin:$PATH
export PYTHONPATH=/gpfs/home/ajb/Code/tsml-eval
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled WANDB_DISABLED=true DO_NOT_TRACK=1
trap 'rc=$?; printf "%s\\n" "$rc" > "$ROOT/preflight-exit-code.txt"' EXIT
"$PY" -u "$ROOT/shape_preflight.py" --root "$ROOT"
"$PY" -u "$ROOT/slurm/tsml_slurm.py" check "$ROOT/experiment.toml"
"$PY" -u "$ROOT/slurm/tsml_slurm.py" start "$ROOT/experiment.toml"
'''
    (root/'preflight.sbatch').write_text(script,encoding='utf-8',newline='\n')
    with tarfile.open(root/'payload.tar','w') as archive:
        for p in root.rglob('*'):
            if p.is_file() and p.name!='payload.tar' and '__pycache__' not in p.parts:
                archive.add(p,arcname=str(p.relative_to(root)).replace('\\','/'),recursive=False)
    return root


if __name__=='__main__':print(prepare())
