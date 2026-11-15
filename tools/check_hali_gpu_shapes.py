"""GPU-node shape preflight for the new-dataset tasks; no model fitting."""
import argparse
import gc
import json
from pathlib import Path
import sys


def run(root):
    root=Path(root);sys.path.insert(0,str(root/'slurm'))
    import tsml_slurm as ts
    e=ts.load_experiment(root/'experiment.toml');ts.read_datasets(e)
    sys.path.insert(0,str(e.repo_dir))
    import numpy as np
    import torch
    from tsml_eval.experiments import get_classifier_by_name
    from aeon_neuro.classification.experimental._neuralbench import NeuralBenchPreprocessor
    from aeon_neuro.classification.deep_learning import BraindecodeClassifier
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable in preflight allocation')
    torch.set_num_threads(2);shapes={}
    for name in e.datasets:
        fields={}
        with (e.data_dir/name/(name+'_TRAIN.ts')).open() as f:
            for line in f:
                parts=line.strip().split()
                if not parts:continue
                if parts[0].lower()=='@data':break
                fields[parts[0].lower()]=parts[1:]
        shapes[name]=(int((fields.get('@dimensions') or fields['@dimension'])[0]),
                      int(fields['@serieslength'][0]),len(fields['@classlabel'])-1)
    report=dict(passed=[],blocked=[],scope='Constructor and zero-input forward only; no fitting or changed padding')
    for task in ts.all_tasks(e):
        if ts.is_complete(e,task):continue
        model=network=None
        try:
            kwargs=ts.task_kwargs(e,task)
            wrapper=get_classifier_by_name(task.estimator,random_state=0,**kwargs)
            channels,points,classes=shapes[task.dataset]
            target=NeuralBenchPreprocessor(sfreq=float(kwargs['sfreq'])).output_sfreq
            processed=round(points*target/float(kwargs['sfreq']))
            x=wrapper._pad(np.zeros((2,channels,processed),dtype=np.float32))
            model=BraindecodeClassifier(model=wrapper.model,model_params=wrapper.model_params,
                                       sfreq=target,device='cuda')
            model.n_classes_=classes
            network=model._build_network(channels,x.shape[-1]).to('cuda').eval()
            with torch.no_grad():out=model._coerce_logits(network(torch.from_numpy(x).to('cuda')))
            if tuple(out.shape)!=(2,classes):raise ValueError('Output shape mismatch')
            report['passed'].append(dict(key=task.key,shape=list(x.shape),classes=classes))
            print('PASS',task.key,flush=True)
        except Exception as exc:
            if isinstance(exc,ImportError) or 'out of memory' in str(exc).lower():raise
            report['blocked'].append(dict(key=task.key,error=repr(exc)))
            print('BLOCK',task.key,repr(exc),flush=True)
        finally:
            del network,model;gc.collect();torch.cuda.empty_cache()
        (root/'shape-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    # Re-load the unchanged recipe after appending explicit task exclusions.
    source=(root/'experiment.toml').read_text()
    exclusions=[x['key'] for x in report['blocked']]
    if exclusions:
        source=source.replace('[experiment]\n','[experiment]\nexclude_tasks = '+json.dumps(exclusions)+'\n',1)
        (root/'experiment.toml').write_text(source)
    report['status']='complete'
    (root/'shape-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise RuntimeError('No compatible tasks')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();run(a.root)
