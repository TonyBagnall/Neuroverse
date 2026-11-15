"""Train the user-selected MRHydra and QUANT locally; no Codabench uploads."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from tools.competition_track2 import DEFAULT_OUTPUT,ROOT,dump,score,table


def native_scores(classifier,name,x):
    if name=='QUANT':return classifier.predict_proba(x)
    hydra=classifier._scale_hydra.transform(classifier._transform_hydra.transform(x))
    rocket=classifier._scale_multirocket.transform(classifier._transform_multirocket.transform(x))
    decisions=classifier.classifier.decision_function(np.concatenate((hydra,rocket),axis=1))
    # Ridge fits{-1,+1} one-vs-rest targets. These are probability-like scores,
    # not calibrated posterior probabilities; no validation labels fit them.
    values=np.clip((decisions+1)/2,0,1)
    return values/np.maximum(values.sum(axis=1,keepdims=True),1e-20)


def train(data=DEFAULT_OUTPUT,output=None,jobs=8):
    import torch,aeon,sklearn
    from aeon.classification.convolution_based import MultiRocketHydraClassifier
    from aeon.classification.interval_based import QUANTClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    torch.set_num_threads(jobs)
    folder=Path(data)/'prepared';out=Path(output or Path(data)/'selected-models');out.mkdir(parents=True,exist_ok=True)
    x=np.load(folder/'X.npy',mmap_mode='r');y=np.load(folder/'y.npy');splits=json.loads((folder/'splits.json').read_text());rows=table(folder/'cases.tsv')
    results={};validation_scores={}
    for name in ['MRHydra','QUANT']:
        if name=='MRHydra':make=lambda:MultiRocketHydraClassifier(n_jobs=jobs,random_state=33)
        else:make=lambda:QUANTClassifier(random_state=33,estimator=ExtraTreesClassifier(n_estimators=200,max_features=.1,criterion='entropy',n_jobs=jobs,random_state=33))
        results[name]={}
        for phase,indices in [('development',splits['train']),('all-public',list(range(len(y))))]:
            path=out/name/phase;path.mkdir(parents=True,exist_ok=True);model_file=path/'model.joblib'
            if model_file.exists():
                raise FileExistsError('Existing fitted model: review receipt before reuse '+str(model_file))
            print('Fitting',name,phase,len(indices),'cases',flush=True)
            start=time.perf_counter();model=make().fit(x[indices],y[indices]);seconds=time.perf_counter()-start
            joblib.dump(model,model_file,compress=3)
            report=dict(classifier=name,phase=phase,train_cases=len(indices),fit_seconds=seconds,random_state=33,n_jobs=jobs,
                aeon=aeon.__version__,sklearn=sklearn.__version__,torch=torch.__version__,path=str(model_file))
            if phase=='development':
                valid=splits['val'];start=time.perf_counter();values=native_scores(model,name,x[valid]);predictions=np.argmax(values,axis=1)
                reference=model.predict(x[valid][:12])
                if not np.array_equal(reference,predictions[:12]):raise ValueError('Derived score labels disagree with aeon')
                validation_scores[name]=values;np.save(path/'validation_scores.npy',values,allow_pickle=False)
                report['validation']=score(y[valid],predictions,[rows[i] for i in valid]);report['validation_seconds']=time.perf_counter()-start
            dump(path/'training.json',report);results[name][phase]=report
            dump(ROOT/'docs/competition-track2-selected-training.json',results)
            print(name,phase,'fit_seconds',seconds,'validation',report.get('validation',{}).get('balanced_accuracy_cell_mean'),flush=True)
            del model
    ensemble=(validation_scores['MRHydra']+validation_scores['QUANT'])/2
    results['equal_weight_ensemble_validation']=score(y[splits['val']],np.argmax(ensemble,axis=1),[rows[i] for i in splits['val']])
    results['ensemble_definition']='Equal mean of QUANT ExtraTrees probabilities and clipped/normalised MRHydra ridge scores; no calibration parameters fitted on validation.'
    dump(ROOT/'docs/competition-track2-selected-training.json',results)
    print('Validation ensemble',results['equal_weight_ensemble_validation']['balanced_accuracy_cell_mean'],flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,default=DEFAULT_OUTPUT);parser.add_argument('--output',type=Path);parser.add_argument('--jobs',type=int,default=8);args=parser.parse_args();train(args.data,args.output,args.jobs)
