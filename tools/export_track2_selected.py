"""Portable export/parity/replay of fitted MRHydra and QUANT; no upload."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import zipfile

import joblib
import numpy as np
import torch
from tools.competition_track2 import ROOT,DEFAULT_OUTPUT,dump,table,score
from tools.build_aeon_track2_submission import official_replay


def parameters(mr,quant,channel_names,mode):
    p=dict(channel_names=np.asarray(channel_names),mode=np.asarray(mode))
    if mode in ['MRHydra','ensemble']:
        hydra=mr._transform_hydra._hydra
        p.update(hydra_weights=hydra.W.numpy(),hydra_index=torch.stack(hydra.idx).numpy(),hydra_h=np.array(hydra.h),hydra_k=np.array(hydra.k),
            hydra_dilations=hydra.dilations.numpy(),hydra_mu=mr._scale_hydra.mu.numpy(),hydra_sigma=mr._scale_hydra.sigma.numpy(),
            mr_mean=mr._scale_multirocket.mean_,mr_scale=mr._scale_multirocket.scale_,ridge_coef=mr.classifier.coef_,ridge_intercept=mr.classifier.intercept_)
        if mr._transform_multirocket.normalise:raise ValueError('Input-normalised MultiROCKET not implemented')
        for prefix,values in [('mr_',mr._transform_multirocket.parameter),('diff_',mr._transform_multirocket.parameter1)]:
            for name,value in zip(['counts','channels','dilations','features','biases'],values,strict=True):p[prefix+name]=value
    if mode in ['QUANT','ensemble']:
        p['quant_divisor']=np.array(quant._transformer.quantile_divisor)
        for i,intervals in enumerate(quant._transformer.intervals_):p['quant_intervals_'+str(i)]=intervals.numpy()
        p['tree_count']=np.array(len(quant._estimator.estimators_))
        for i,estimator in enumerate(quant._estimator.estimators_):
            tree=estimator.tree_;prefix='tree_'+str(i)+'_'
            for name,value in [('left',tree.children_left),('right',tree.children_right),('feature',tree.feature),('threshold',tree.threshold),('value',tree.value[:,0,:])]:p[prefix+name]=value
    return p


def load_portable(folder,channel_names):
    path=Path(folder)/'submission.py';spec=importlib.util.spec_from_file_location('selected_submission',path);module=importlib.util.module_from_spec(spec)
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),module.__dict__)
    model=module.Solver().load_model(dict(submission_dir=Path(folder),device='cpu',n_chans=41,n_times=480,sfreq=120,n_classes=3,ch_names=channel_names))
    return model,module


def feature_checks(mr,quant,model,module,x):
    reports={}
    if model.mode in ['MRHydra','ensemble']:
        a=mr._scale_hydra.transform(mr._transform_hydra.transform(x)).numpy();b=module.hydra(x,model.p)
        reports['hydra_max_abs_error']=float(np.max(np.abs(a-b)))
        if not np.allclose(a,b,rtol=1e-5,atol=2e-3):raise ValueError('Hydra feature parity failed '+str(reports))
        a=mr._transform_multirocket.transform(x);b=module.multi_rocket(x,model.p)
        reports['multirocket_max_abs_error']=float(np.max(np.abs(a-b)))
        if not np.allclose(a,b,rtol=1e-5,atol=2e-3):raise ValueError('MultiROCKET feature parity failed '+str(reports))
    if model.mode in ['QUANT','ensemble']:
        a=quant._transformer.transform(x);b=module.quant_features(x,model.p)
        reports['quant_feature_max_abs_error']=float(np.max(np.abs(a-b)))
        if not np.array_equal(a,b):raise ValueError('QUANT feature parity failed '+str(reports))
        a=quant._estimator.predict_proba(a);b=module.forest_probabilities(b,model.p)
        reports['forest_probability_max_abs_error']=float(np.max(np.abs(a-b)))
        if not np.allclose(a,b,rtol=0,atol=1e-12):raise ValueError('Forest probability parity failed '+str(reports))
    return reports


def export(data=DEFAULT_OUTPUT,output=None,phase='development',limit=None,replay=True,modes=None):
    torch.set_num_threads(1)
    root=Path(data);folder=root/'prepared';source=root/'selected-models';out=Path(output or root/'selected-submissions')/phase;out.mkdir(parents=True,exist_ok=True)
    x=np.load(folder/'X.npy',mmap_mode='r');y=np.load(folder/'y.npy');rows=table(folder/'cases.tsv');split=json.loads((folder/'splits.json').read_text());metadata=json.loads((folder/'metadata.json').read_text())
    modes=modes or ['MRHydra','QUANT','ensemble']
    mr=joblib.load(source/'MRHydra'/phase/'model.joblib') if any(m in ['MRHydra','ensemble'] for m in modes) else None
    quant=joblib.load(source/'QUANT'/phase/'model.joblib') if any(m in ['QUANT','ensemble'] for m in modes) else None
    check=split['val'] if phase=='development' else list(range(len(y)))
    if limit is not None:check=np.asarray(check)[np.linspace(0,len(check)-1,min(limit,len(check)),dtype=int)].tolist()
    from tools.train_track2_selected import native_scores
    fitted=[(name,clf) for name,clf in [('MRHydra',mr),('QUANT',quant)] if clf is not None]
    reference={name:np.concatenate([native_scores(clf,name,x[check[i:i+32]]) for i in range(0,len(check),32)]) for name,clf in fitted}
    hard_reference={name:np.concatenate([clf.predict(x[check[i:i+32]]) for i in range(0,len(check),32)]) for name,clf in fitted}
    report_path=ROOT/'docs'/('competition-track2-selected-export-'+phase+('-smoke' if limit is not None else '')+'.json');reports={}
    for mode in modes:
        destination=out/mode;destination.mkdir(exist_ok=False)
        np.savez_compressed(destination/'weights.npz',**parameters(mr,quant,metadata['channel_names'],mode))
        shutil.copy2(ROOT/'tools/selected_track2_submission.py',destination/'submission.py')
        # Same aeon licence as the already tested prototype export.
        import importlib.metadata
        distribution=importlib.metadata.distribution('aeon');licence=next(distribution.locate_file(f) for f in distribution.files if str(f).endswith('licenses/LICENSE'));shutil.copy2(licence,destination/'LICENSE.txt')
        model,module=load_portable(destination,metadata['channel_names']);features=feature_checks(mr,quant,model,module,x[check[:2]])
        expected=(reference['MRHydra']+reference['QUANT'])/2 if mode=='ensemble' else reference[mode]
        predictions=[];timings=[];max_score_error=0.0
        for number,i in enumerate(check):
            start=time.perf_counter();values=model.scores(torch.tensor(x[i:i+1],dtype=torch.float32));timings.append(time.perf_counter()-start)
            predictions.append(int(np.argmax(values,axis=1)[0]));max_score_error=max(max_score_error,float(np.max(np.abs(values[0]-expected[number]))))
            if (number+1)%100==0:print(phase,mode,'portable parity',number+1,'/',len(check),flush=True)
        predictions=np.asarray(predictions)
        expected_labels=np.argmax(expected,axis=1) if mode=='ensemble' else hard_reference[mode]
        if not np.array_equal(predictions,expected_labels):raise ValueError(mode+' portable predictions differ on '+str(np.count_nonzero(predictions!=expected_labels))+' cases')
        result=dict(mode=mode,phase=phase,verified_cases=len(check),class_predictions_identical=True,max_score_error=max_score_error,
            verification_case_indices=check,verification_scope='All development validation cases' if limit is None and phase=='development' else ('All public cases' if limit is None else 'Evenly spaced sample of public/development cases'),
            median_seconds_per_window=float(np.median(timings)),p95_seconds_per_window=float(np.quantile(timings,.95)),total_prediction_seconds=float(sum(timings)),
            features=features,aeon_inference_required=False,sklearn_inference_required=False,numba_inference_required=False,
            training_data='Public PROTEUS nm000290v1.0.0',training_cases=1521 if phase=='development' else len(y))
        if phase=='development':
            result['validation_score']=score(y[check],predictions,[rows[i] for i in check])
            if replay:result['official_replay']=official_replay(model,x[check],y[check],[rows[i] for i in check])
        dump(destination/'method.json',result)
        zipped=out/(mode+'.zip')
        with zipfile.ZipFile(zipped,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(destination.iterdir()):
                if p.is_file():z.write(p,p.name)
        result.update(zip_path=str(zipped),zip_bytes=zipped.stat().st_size,zip_sha256=hashlib.sha256(zipped.read_bytes()).hexdigest())
        reports[mode]=result;dump(report_path,reports)
        print(phase,mode,'verified',len(check),'zip_bytes',result['zip_bytes'],flush=True)
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['development','all-public'],default='development');parser.add_argument('--output',type=Path);parser.add_argument('--limit',type=int);parser.add_argument('--no-replay',action='store_true');args=parser.parse_args();export(output=args.output,phase=args.phase,limit=args.limit,replay=not args.no_replay)
