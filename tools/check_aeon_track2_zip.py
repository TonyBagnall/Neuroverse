"""Fresh-process replay: block aeon/numba and load only the submitted ZIP.

Run with python -I. The supplied runtime/kit directories provide the official
framework, not model code. No training or model-specific source path is added.
"""
import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import types
import zipfile


class BlockTrainingLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'aeon','numba'}:raise ImportError('Training library intentionally unavailable: '+fullname)
        return None


def check(zip_path,data,kit,runtime,temp_root):
    sys.meta_path.insert(0,BlockTrainingLibraries())
    sys.path.insert(0,str(runtime));sys.path.insert(0,str(kit))
    import numpy as np
    import torch
    with tempfile.TemporaryDirectory(dir=temp_root,prefix='isolated-submission-') as tmp:
        folder=Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            names=z.namelist()
            if set(names)!={'submission.py','weights.npz','LICENSE.txt','method.json'}:raise ValueError('Unexpected ZIP layout')
            if any(Path(n).name!=n for n in names):raise ValueError('Nested or unsafe ZIP path')
            z.extractall(folder)
        def snapshot():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
        before=snapshot();module=types.ModuleType('submission');module.__file__=str(folder/'submission.py')
        exec(compile((folder/'submission.py').read_text(encoding='utf-8'),module.__file__,'exec'),module.__dict__)
        with np.load(folder/'weights.npz',allow_pickle=False) as weights:channels=weights['channel_names'].tolist()
        model=module.Solver().load_model(dict(submission_dir=folder,device='cpu',n_chans=41,n_times=480,sfreq=120,n_classes=3,ch_names=channels))
        x=np.load(Path(data)/'prepared/X.npy',mmap_mode='r');indices=np.linspace(0,len(x)-1,12,dtype=int)
        predictions=[int(model.predict(torch.tensor(x[i:i+1],dtype=torch.float32))[0]) for i in indices]
        if not set(predictions)<=set(range(3)):raise ValueError('Invalid predictions')
        if snapshot()!=before or len(list(folder.iterdir()))!=4:raise ValueError('Submission wrote into read-only artifact layout')
        if any(k.split('.')[0] in {'aeon','numba'} for k in sys.modules):raise ValueError('Training dependency unexpectedly imported')
        return dict(zip=str(zip_path),zip_sha256=hashlib.sha256(Path(zip_path).read_bytes()).hexdigest(),passed=True,
            aeon_imports_blocked=True,numba_imports_blocked=True,cases=len(indices),predictions=predictions,
            torch_input=True,submission_artifacts_unchanged=True,additional_python_modules=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--zip',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--kit',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--temp-root',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    result=check(a.zip,a.data,a.kit,a.runtime,a.temp_root);a.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
