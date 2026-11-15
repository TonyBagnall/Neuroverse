"""Train aeon MiniROCKET, export portable inference, verify and create ZIPs.

Does not submit anything. All trained artifacts remain outside the repository.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from tools.competition_track2 import ROOT,DEFAULT_OUTPUT,dump,table,score


def export(classifier,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    counts,channels,dilations,features,biases=classifier._transformer.parameters
    np.savez_compressed(folder/'weights.npz',counts=counts,channel_indices=channels,dilations=dilations,features_per_dilation=features,biases=biases,
        scale=classifier._scaler.scale_,coef=classifier._estimator.coef_,intercept=classifier._estimator.intercept_,classes=classifier.classes_,
        input_shape=np.asarray([classifier.n_channels_,classifier.n_timepoints_],dtype=np.int32))
    shutil.copy2(ROOT/'tools/aeon_track2_submission.py',folder/'submission.py')
    import importlib.metadata
    distribution=importlib.metadata.distribution('aeon')
    licence=next((distribution.locate_file(f) for f in distribution.files if str(f).endswith('licenses/LICENSE')),None)
    if licence is None:raise RuntimeError('Cannot export without aeon licence')
    shutil.copy2(licence,folder/'LICENSE.txt')
    return folder


def load_submission(folder):
    path=Path(folder)/'submission.py'
    spec=importlib.util.spec_from_file_location('submission',path)
    module=importlib.util.module_from_spec(spec)
    # Match read-only submission mounting: do not create a bytecode cache.
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),module.__dict__)
    with np.load(Path(folder)/'weights.npz',allow_pickle=False) as data:
        names=data['channel_names'].tolist() if 'channel_names' in data else []
    return module.Solver().load_model(dict(submission_dir=Path(folder),n_chans=41,n_times=480,n_classes=3,sfreq=120,device='cpu',ch_names=names))


def finalize(data=DEFAULT_OUTPUT,output=None):
    """Include the exact sensor contract and refresh ZIP hashes after validation."""
    out=Path(output or Path(data)/'aeon-submissions')
    metadata=json.loads((Path(data)/'prepared/metadata.json').read_text(encoding='utf-8'))
    results=json.loads((ROOT/'docs/competition-track2-aeon-submission-results.json').read_text(encoding='utf-8'))
    for name in ['development','all-public']:
        folder=out/name;weights=folder/'weights.npz'
        with np.load(weights,allow_pickle=False) as data_file:values={k:data_file[k] for k in data_file.files}
        values['channel_names']=np.asarray(metadata['channel_names'])
        pending=folder/'weights.pending'
        with pending.open('wb') as f:np.savez_compressed(f,**values)
        pending.replace(weights)
        shutil.copy2(ROOT/'tools/aeon_track2_submission.py',folder/'submission.py')
        method=json.loads((folder/'method.json').read_text(encoding='utf-8'))
        method['expected_channel_names']=metadata['channel_names'];method['channel_order_checked_at_load']=True
        dump(folder/'method.json',method)
        zipped=out/(name+'.zip')
        with zipfile.ZipFile(zipped,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(folder.iterdir()):
                if path.is_file():z.write(path,path.name)
        results[name].update(expected_channel_names=metadata['channel_names'],channel_order_checked_at_load=True,
            zip_bytes=zipped.stat().st_size,zip_sha256=hashlib.sha256(zipped.read_bytes()).hexdigest())
    dump(ROOT/'docs/competition-track2-aeon-submission-results.json',results)
    return results


def parity(classifier,model,x,indices):
    """Compare all streamed class predictions plus transformed-feature error."""
    expected=classifier.predict(x[indices]);predictions=[];timings=[]
    for i in indices:
        start=time.perf_counter();predictions.append(int(model.predict(x[i:i+1])[0]));timings.append(time.perf_counter()-start)
        if len(predictions)%250==0:print('Portable parity',len(predictions),'/',len(indices),flush=True)
    predictions=np.asarray(predictions)
    mismatches=int(np.count_nonzero(expected!=predictions))
    if mismatches:raise ValueError(f'Portable class predictions differ from aeon on{mismatches}/{len(indices)}cases')
    sample=indices[:min(12,len(indices))]
    a=classifier._transformer.transform(x[sample]);b=model.features(x[sample])
    return dict(cases=len(indices),class_predictions_identical=True,feature_sample_cases=len(sample),feature_max_abs_error=float(np.max(np.abs(a-b))),
        feature_unequal_values=int(np.count_nonzero(a!=b)),median_seconds_per_window=float(np.median(timings)),p95_seconds_per_window=float(np.quantile(timings,.95)),total_prediction_seconds=float(sum(timings))),predictions


def official_replay(model,x,y,rows):
    """Execute the organiser's actual Objective and streaming implementation."""
    import sys
    kit=ROOT/'.openeegbench/competition-17982/kit'
    sys.path.insert(0,str(kit))
    from benchmark_utils.data import make_loader
    spec=importlib.util.spec_from_file_location('track2_objective',kit/'tracks/bci_decoding/objective.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def encode(keys):
        mapping={};return np.asarray([mapping.setdefault(k,len(mapping)) for k in keys])
    loader=make_loader(x,y,batch_size=16,device='cpu',record_id=encode([r['source_file'] for r in rows]),
        stream_id=encode([(r['subject'],r['session']) for r in rows]),context_id=encode([r['interface'] for r in rows]),
        onset=np.asarray([float(r['cue_onset_seconds']) for r in rows]))
    objective=module.Objective();objective.set_data(None,loader,3)
    return objective.evaluate_result(model)


def build(data=DEFAULT_OUTPUT,output=None,kernels=10000,jobs=8):
    from aeon.classification.convolution_based import MiniRocketClassifier
    folder=Path(data)/'prepared';out=Path(output or Path(data)/'aeon-submissions');out.mkdir(parents=True,exist_ok=True)
    x=np.load(folder/'X.npy',mmap_mode='r');y=np.load(folder/'y.npy');rows=table(folder/'cases.tsv');splits=json.loads((folder/'splits.json').read_text())
    results={}
    for name,train,check in [('development',splits['train'],splits['val']),('all-public',list(range(len(y))),list(range(len(y))))]:
        print('Fitting',name,'aeon MiniROCKET',len(train),'cases',kernels,'kernels',flush=True)
        start=time.perf_counter();classifier=MiniRocketClassifier(n_kernels=kernels,n_jobs=jobs,random_state=33)
        classifier.fit(x[train],y[train]);fit_seconds=time.perf_counter()-start
        destination=export(classifier,out/name);model=load_submission(destination)
        agreement,predictions=parity(classifier,model,x,check)
        method=dict(model='aeon MiniRocketClassifier, portable NumPy inference',aeon_version='1.5.0',requested_kernels=kernels,actual_features=len(classifier._transformer.parameters[-1]),
            random_state=33,train_cases=len(train),training_split=name,training_data='Public PROTEUS nm000290v1.0.0 only',fit_seconds=fit_seconds,
            inference_dependencies=['numpy','benchmark_utils.base_solver supplied by competition'],no_runtime_installation=True,aeon_import_required=False,
            parity=agreement,limitations='NumPy versus Numba floating arithmetic may differ near learned thresholds; class agreement verified on stated cases, not a proof for every possible future input. Unseen sealed data inaccessible.')
        if name=='development':
            rr=[rows[i] for i in check];method['validation_score']=score(y[check],predictions,rr)
            method['official_objective_replay']=official_replay(model,x[check],y[check],rr)
        dump(destination/'method.json',method)
        zipped=out/(name+'.zip')
        with zipfile.ZipFile(zipped,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(destination.iterdir()):
                if p.is_file():z.write(p,p.name)
        method['zip_bytes']=zipped.stat().st_size;method['zip_sha256']=hashlib.sha256(zipped.read_bytes()).hexdigest();method['zip_path']=str(zipped)
        results[name]=method;dump(ROOT/'docs/competition-track2-aeon-submission-results.json',results)
        print(name,'fit_seconds',fit_seconds,'parity',agreement,'zip',zipped.stat().st_size,flush=True)
    return finalize(data,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,default=DEFAULT_OUTPUT);parser.add_argument('--output',type=Path);parser.add_argument('--kernels',type=int,default=10000);parser.add_argument('--jobs',type=int,default=8);args=parser.parse_args();build(args.data,args.output,args.kernels,args.jobs)
