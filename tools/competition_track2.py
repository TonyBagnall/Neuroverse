"""Download/format public PROTEUS for Codabench17982 local classification.

Independent of archive registration. Never downloads sealed evaluation data.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MANIFEST_URL='https://data.nemar.org/nm000290/v1.0.0/manifest.json'
MANIFEST_SHA='646d0f1347599c69b77ca9cc9b673cf80c2100c82ec77d3c9f1959ff9e2e61bb'
KIT_COMMIT='7160f93ae5e86249e7ee5ce0583e34ec7163c521'
LABELS={'mi':0,'sub':1,'word':2}
CODES={'mi':33024,'sub':33025,'word':33026}
TASKS={'AcquisitionBH':'BrainHero','OnlineRawBH':'BrainHero',
       'AcquisitionGraz':'Graz','OnlineRawGraz':'Graz'}
DEFAULT_CACHE=Path('D:/Data/EEGSources/Competitions/NeuralInterfaces2026Track2')
DEFAULT_OUTPUT=Path('D:/Data/EEGCompetitions/NeuralInterfaces2026Track2')


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def numeric(value):
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,np.ndarray):return value.tolist()
        raise TypeError(f'Unsupported JSON value: {type(value).__name__}')
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=numeric)+'\n',encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def table(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))


def source_digest(path,algorithm):
    if algorithm=='git':
        raw=Path(path).read_bytes()
        return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if algorithm!='sha256':raise ValueError('Unreviewed source checksum algorithm')
    return sha(path)


def safe_target(root,relative):
    target=(Path(root)/relative).resolve()
    if not target.is_relative_to(Path(root).resolve()):raise ValueError('Unsafe source path')
    return target


def transfer_mode(status,content_range,offset,total):
    """Append only when the server confirms the exact requested byte range."""
    if status==200:return 'wb'
    if status==206:
        match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',content_range or '')
        if match and int(match[1])==offset and int(match[3])==total and int(match[2])==total-1:
            return 'ab' if offset else 'wb'
    raise ValueError('Unexpected download range response')


def download(cache=DEFAULT_CACHE,pilot=False):
    import requests
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    manifest=cache/'manifest.json'
    if not manifest.exists():
        response=requests.get(MANIFEST_URL,timeout=60);response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest()!=MANIFEST_SHA:raise ValueError('Published manifest changed: review rather than repin silently')
        manifest.write_bytes(response.content)
    if sha(manifest)!=MANIFEST_SHA:raise ValueError('Source manifest changed')
    files=json.loads(manifest.read_text(encoding='utf-8'))
    def selected(entry):
        name=entry['path']
        if name.startswith('.'):return False
        if not name.endswith('.edf'):return True
        return any('_task-'+task+'_' in name for task in TASKS) and (not pilot or 'sub-P01/ses-1/' in name and 'AcquisitionBH' in name)
    def one(entry):
        path=safe_target(cache,entry['path'])
        def valid(p):return p.exists() and p.stat().st_size==entry['size'] and source_digest(p,entry['checksum_algorithm'])==entry['checksum']
        if not valid(path):
            path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_name(path.name+'.partial')
            last=None
            for attempt in range(3):
                try:
                    offset=pending.stat().st_size if pending.exists() else 0
                    if not 0<offset<entry['size']:offset=0
                    headers={'Range':f'bytes={offset}-'} if offset else {}
                    with requests.get(entry['url'],headers=headers,stream=True,timeout=(20,120)) as response:
                        response.raise_for_status()
                        mode=transfer_mode(response.status_code,response.headers.get('Content-Range'),offset,entry['size'])
                        with pending.open(mode) as f:
                            for block in response.iter_content(1024*1024):f.write(block)
                    if not valid(pending):raise ValueError('Downloaded source checksum mismatch '+entry['path'])
                    pending.replace(path);last=None;break
                except (requests.RequestException,ValueError) as exc:last=exc
            if last:raise last
        return dict(entry,sha256=sha(path))
    entries=[f for f in files if selected(f)]
    with ThreadPoolExecutor(6) as pool:receipt=list(pool.map(one,entries))
    description=json.loads((cache/'dataset_description.json').read_text(encoding='utf-8'))
    if description['License']!='CC-BY-4.0':raise ValueError('Original data licence changed')
    dump(cache/('receipt-pilot.json' if pilot else 'receipt-full.json'),dict(version='1.0.0',manifest_sha256=MANIFEST_SHA,licence=description['License'],files=receipt))
    print('Verified source files',len(receipt),'bytes',sum(f['size'] for f in receipt),flush=True)
    return receipt


def development_split(subjects):
    """Pinned starter's two-stage60/20/20 subject split, seed33."""
    from sklearn.model_selection import train_test_split
    ordered=sorted(set(subjects))
    train_val,test=train_test_split(ordered,test_size=.2,random_state=33)
    train,val=train_test_split(train_val,test_size=.2/.8,random_state=33)
    return {p:split for split,people in [('train',train),('val',val),('test',test)] for p in people}


def cue_windows(events,native_samples,hz=500):
    out=[]
    for event_index,event in enumerate(events):
        label=event['trial_type']
        if label not in LABELS:continue
        onset=float(event['onset']);sample=int(event['sample'])
        if int(event['value'])!=CODES[label] or abs(sample-onset*hz)>.50001:raise ValueError('Conflicting cue code or native sample grid')
        start=sample+round(hz);end=start+round(4*hz)
        if start<0 or end>native_samples:raise ValueError('Incomplete cue window; source review required')
        out.append(dict(event_index=event_index,source_label=label,label=LABELS[label],cue_onset_seconds=onset,
                        cue_sample=sample,start_sample=start,end_sample=end,
                        start_time_seconds=onset+1,end_time_seconds=onset+5,
                        processed_start_sample=round((onset+1)*120)))
    return out


def preprocess(raw):
    """Mirror NeuralSet0.4 MneRaw preprocessing, including notch harmonics.

    Same float32-SI cache boundary, then float32 microvolt scaling; float64
    export preserves those model-facing values. Full harness parity unclaimed.
    """
    raw.load_data()
    hz=raw.info['sfreq']
    notches=[f for base in [50,60] for f in np.arange(base,min(hz/2,301),base)]
    raw.notch_filter(notches,picks='all',phase='zero',filter_length='auto',n_jobs=1,verbose=False)
    raw.filter(.1,75,picks='all',skip_by_annotation=('edge','bad_acq_skip'),n_jobs=1,verbose=False)
    raw.resample(120,n_jobs=1,verbose=False)
    return raw.get_data().astype(np.float32)*np.float32(1e6)


def write_ts(path,x,y):
    with Path(path).open('w',encoding='utf-8',newline='\n') as f:
        f.write('@problemName ProteusCommands\n@timestamps false\n@missing false\n@univariate false\n@dimensions 41\n@equalLength true\n@seriesLength 480\n@classLabel true 0 1 2\n@data\n')
        for case,label in zip(x,y,strict=True):
            f.write(':'.join(','.join(format(float(v),'.17g') for v in channel) for channel in case)+':'+str(label)+'\n')


def write_rows(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)


def prepare(cache=DEFAULT_CACHE,output=DEFAULT_OUTPUT,pilot=False):
    import mne
    cache=Path(cache);output=Path(output)/('pilot' if pilot else 'prepared')
    if output.exists():raise FileExistsError(output)
    receipt=json.loads((cache/('receipt-pilot.json' if pilot else 'receipt-full.json')).read_text(encoding='utf-8'))
    for entry in receipt['files']:
        path=cache/entry['path']
        if path.stat().st_size!=entry['size'] or sha(path)!=entry['sha256']:raise ValueError('Cached source changed')
    chunks=[];rows=[];recordings=[];channel_names=None;sensors=None;seen={}
    edfs=[e for e in receipt['files'] if e['path'].endswith('.edf')]
    def order(e):
        name=e['path'];return (re.search(r'sub-([^/]+)',name)[1],int(re.search(r'ses-([^/]+)',name)[1]),int(re.search(r'_run-(\d+)',name)[1]))
    for number,entry in enumerate(sorted(edfs,key=order)):
        name=entry['path'];path=cache/name;prefix=str(path)[:-7]
        channels=table(prefix+'channels.tsv');events=table(prefix+'events.tsv')
        bids=json.loads(Path(prefix+'eeg.json').read_text(encoding='utf-8'))
        raw=mne.io.read_raw_edf(path,preload=False,verbose=False)
        names=[r['name'] for r in channels]
        if raw.ch_names!=names or len(names)!=41 or any(c['type']!='EEG' or c['units'] not in ['µV','uV'] for c in channels):raise ValueError('Native/BIDS channel identity mismatch')
        if raw.info['sfreq']!=500 or bids['SamplingFrequency']!=500:raise ValueError('Source frequency mismatch')
        if channel_names is None:channel_names=names;sensors=channels
        if channel_names!=names:raise ValueError('Channel order differs: no silent reordering')
        windows=cue_windows(events,raw.n_times)
        # Independently corroborate every command cue against native EDF annotations.
        annotations=list(zip(raw.annotations.onset,raw.annotations.description))
        for cue in windows:
            if not any(abs(float(t)-cue['cue_onset_seconds'])<=.00201 and str(desc) in [cue['source_label'],str(CODES[cue['source_label']])] for t,desc in annotations):
                raise ValueError('BIDS/native cue annotation mismatch '+name)
        person,session,run=order(entry);task=re.search(r'_task-([^_]+)',name)[1]
        recording=dict(source_file=name,sha256=entry['sha256'],subject=person,session=str(session),run=str(run),task=task,interface=TASKS[task],
            native_sampling_frequency_hz=500,native_samples=raw.n_times,native_bids=bids,native_bids_channels=channels,native_units=raw._orig_units,
            annotations=[dict(onset=float(t),duration=float(d),description=str(s)) for t,d,s in zip(raw.annotations.onset,raw.annotations.duration,raw.annotations.description)])
        signal=preprocess(raw);cases=[]
        for cue in windows:
            start=cue['processed_start_sample'];case=np.asarray(signal[:,start:start+480],dtype='<f8')
            if case.shape!=(41,480) or not np.isfinite(case).all():raise ValueError('Invalid processed case')
            digest=hashlib.sha256(case.tobytes()).hexdigest()
            if digest in seen:raise ValueError('Exact duplicate cases: '+str((seen[digest],name,cue['event_index'])))
            seen[digest]=(name,cue['event_index'])
            rows.append(dict(case_index=len(rows),subject=person,session=str(session),run=str(run),task=task,interface=TASKS[task],
                source_file=name,source_file_sha256=entry['sha256'],source_event_row=cue['event_index'],original_trial_id='',original_partition='public_training_release',
                original_frequency_hz=500,processed_frequency_hz=120,processed_end_sample=start+480,signal_sha256=digest,**cue))
            cases.append(case)
        if cases:chunks.append(np.stack(cases))
        recording['cases']=len(cases);recordings.append(recording)
        print('Processed',number+1,'/',len(edfs),'cases',len(rows),flush=True)
    x=np.concatenate(chunks);y=np.asarray([r['label'] for r in rows],dtype=np.int64)
    partition=development_split(r['subject'] for r in rows) if not pilot else {r['subject']:'train' for r in rows}
    for row in rows:row['development_partition']=partition[row['subject']]
    splits={k:[r['case_index'] for r in rows if r['development_partition']==k] for k in ['train','val','test']}
    # A genuinely chronological cross-session check exists for only2public people.
    sessions=defaultdict(set)
    for row in rows:sessions[row['subject']].add(int(row['session']))
    repeated={p for p,s in sessions.items() if len(s)>1}
    splits['cross_session_train']=[r['case_index'] for r in rows if r['subject'] not in repeated or int(r['session'])<max(sessions[r['subject']])]
    splits['cross_session_test']=[r['case_index'] for r in rows if r['subject'] in repeated and int(r['session'])==max(sessions[r['subject']])]
    output.mkdir(parents=True)
    np.save(output/'X.npy',x,allow_pickle=False);np.save(output/'y.npy',y,allow_pickle=False)
    write_rows(output/'cases.tsv',rows);write_rows(output/'channels.tsv',[dict(array_index=i,**c) for i,c in enumerate(sensors)])
    dump(output/'splits.json',splits)
    validation=dict(shape=list(x.shape),labels=sorted(set(y.tolist())),class_counts=dict(Counter(str(v) for v in y)),
        finite=bool(np.isfinite(x).all()),constant_case_channel_pairs=int((np.ptp(x,axis=-1)==0).sum()),
        effectively_flat_case_channel_pairs=int((x.std(axis=-1)<=1e-7).sum()),duplicate_cases=0,
        channel_order_verified_all_recordings=True,native_edf_bids_cues_verified=True,
        subjects=sorted(sessions),sessions={p:sorted(s) for p,s in sessions.items()},split_counts={k:len(v) for k,v in splits.items()})
    for key in ['X','y']:
        if not np.array_equal(np.load(output/(key+'.npy'),allow_pickle=False),{'X':x,'y':y}[key]):raise ValueError('Array readback mismatch')
    if not pilot:
        if set(r['subject'] for r in rows if r['development_partition']=='test') & set(r['subject'] for r in rows if r['development_partition']!='test'):raise ValueError('Subject leakage')
        for protocol in ['subject_holdout','cross_session']:
            folder=output/protocol;folder.mkdir()
            train=sorted(splits['train']+splits['val']) if protocol=='subject_holdout' else splits['cross_session_train']
            test=splits['test'] if protocol=='subject_holdout' else splits['cross_session_test']
            for label,indices in [('TRAIN',train),('TEST',test)]:
                write_ts(folder/f'ProteusCommands_{label}.ts',x[indices],y[indices]);write_rows(folder/f'cases_{label.lower()}.tsv',[rows[i] for i in indices])
            # Every exported token and case/label order is independently read back.
            from aeon.datasets import load_from_ts_file
            for label,indices in [('TRAIN',train),('TEST',test)]:
                xx,yy=load_from_ts_file(folder/f'ProteusCommands_{label}.ts')
                if not np.array_equal(xx,x[indices]) or not np.array_equal(yy.astype(int),y[indices]):raise ValueError('Exact TS readback failed')
            validation[protocol+'_exact_ts_readback']=True
    # Verify the serialised case table independently against labels and signals.
    stored_rows=table(output/'cases.tsv')
    for index,row in enumerate(stored_rows):
        if int(row['case_index'])!=index or int(row['label'])!=y[index] or row['signal_sha256']!=hashlib.sha256(x[index].tobytes()).hexdigest():raise ValueError('Case sidecar alignment failed')
    if len(stored_rows)!=len(y):raise ValueError('Case sidecar row count mismatch')
    validation['serialised_case_metadata_verified']=True
    metadata=dict(dataset='PROTEUS public Track2',source='nm000290 v1.0.0',source_url='https://nemar.org/dataset/nm000290',
        licence='CC-BY-4.0',manifest_sha256=MANIFEST_SHA,competition_kit_commit=KIT_COMMIT,channel_names=channel_names,label_mapping=LABELS,
        input_contract='N x41x480 at120Hz, numeric microvolts; official online predict receives only1x41x480, no IDs or labels',
        preprocessing='MNE native SI decode; notch50/60Hz and harmonics below Nyquist; zero-phase FIR0.1–75Hz; whole-record resampling120Hz; float32 SI cache then float32 microvolt scale (NeuralSet0.4 boundaries), exported exactly as float64. No learned recording scaling, clipping, rereference, flat-channel removal or label-based rejection.',
        preprocessing_parity='Reviewed against NeuralSet0.4 code; complete NeuralBench harness numerical parity not yet established. MNE version recorded; evaluate local results as local only.',
        epoch='1–5seconds after each original cue;480samples, end-exclusive, round120*(onset+1). Native zero-based interval bounds retained; filter support is whole source recording.',
        split_definition='Development subject split6/2/2 via two sklearn train_test_split calls seed33; train+val merged in subject_holdout TRAIN.ts. Fixed validation indices remain for model selection. Source order is subject/session/numeric run/cue. Cross-session TRAIN uses sessions earlier than each repeated-person final public session plus all other public people; TEST final public sessions of repeated people only. Neither protocol is the sealed competition split.',
        streaming='Deep-copy fitted model per subject/session, reset_state if supplied, predict one case at a time in numeric run/onset order. Do not pass grouping metadata to model; no future-window statistics or cross-session state.',
        scoring='Mean balanced accuracy across subject/session/interface cells; also pooled/session diagnostics. Class recall uses union of true/predicted classes, matching pinned official helpers.',
        versions=dict(mne=mne.__version__,numpy=np.__version__),recordings=recordings,validation=validation)
    dump(output/'metadata.json',metadata)
    evidence=output/'source';evidence.mkdir()
    for name in ['manifest.json','dataset_description.json','participants.tsv','README.md']:
        shutil.copy2(cache/name,evidence/name)
    code=output/'conversion_code';code.mkdir()
    shutil.copy2(Path(__file__),code/Path(__file__).name)
    shutil.copy2(ROOT/'tools/requirements-competition-track2.txt',code/'requirements.txt')
    if (ROOT/'docs/competition-track2.md').exists():shutil.copy2(ROOT/'docs/competition-track2.md',output/'README.md')
    dump(output/'artifacts.json',{p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.rglob('*') if p.is_file()})
    dump(ROOT/'docs'/('competition-track2-'+('pilot' if pilot else 'preparation')+'-result.json'),dict(path=str(output),**validation))
    print(json.dumps(validation,indent=2),flush=True)
    return validation


def balanced_accuracy(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return float(np.mean([np.mean(p[y==c]==c) if np.any(y==c) else 0 for c in np.union1d(y,p)]))


def score(y,p,rows):
    if len(y)!=len(p) or len(rows)!=len(y) or not np.isin(p,[0,1,2]).all():raise ValueError('Invalid predictions/alignment')
    cells=defaultdict(list);sessions=defaultdict(list)
    for i,r in enumerate(rows):cells[(r['subject'],r['session'],r['interface'])].append(i);sessions[(r['subject'],r['session'])].append(i)
    cell_scores=[dict(subject=k[0],session=k[1],interface=k[2],cases=len(v),balanced_accuracy=balanced_accuracy(y[v],p[v])) for k,v in cells.items()]
    return dict(balanced_accuracy_cell_mean=float(np.mean([r['balanced_accuracy'] for r in cell_scores])),balanced_accuracy_session_mean=float(np.mean([balanced_accuracy(y[v],p[v]) for v in sessions.values()])),balanced_accuracy=balanced_accuracy(y,p),accuracy=float(np.mean(y==p)),cells=cell_scores)


def predict_streams(model,x,rows):
    predictions=np.empty(len(rows),dtype=int);groups=defaultdict(list)
    for i,r in enumerate(rows):groups[(r['subject'],r['session'])].append(i)
    for indices in groups.values():
        current=copy.deepcopy(model)
        if hasattr(current,'reset_state'):current.reset_state()
        for i in sorted(indices,key=lambda i:(int(rows[i]['run']),float(rows[i]['cue_onset_seconds']))):
            result=np.asarray(current.predict(x[i:i+1]))
            if result.shape!=(1,) or not np.issubdtype(result.dtype,np.integer) or result[0] not in [0,1,2]:raise ValueError('Invalid streaming model return')
            predictions[i]=result[0]
    return predictions


class BandpowerClassifier:
    """Fixed, untuned CPU baseline; no per-person or test-recording statistics."""
    @staticmethod
    def features(x):
        from scipy.signal import welch
        f,power=welch(x,fs=120,nperseg=240,axis=-1)
        return np.concatenate([np.log(np.maximum(power[...,(f>=lo)&(f<hi)].mean(-1),1e-20)) for lo,hi in [(4,8),(8,13),(13,30),(30,45)]],axis=1)

    def fit(self,x,y):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        self.model=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=33))
        self.model.fit(self.features(x),y);return self

    def predict(self,x):return self.model.predict(self.features(x))


def baseline(output=DEFAULT_OUTPUT):
    folder=Path(output)/'prepared';x=np.load(folder/'X.npy',mmap_mode='r');y=np.load(folder/'y.npy');rows=table(folder/'cases.tsv');splits=json.loads((folder/'splits.json').read_text());results={}
    for name,train,test in [('development_validation',splits['train'],splits['val']),('subject_holdout',sorted(splits['train']+splits['val']),splits['test']),('cross_session',splits['cross_session_train'],splits['cross_session_test'])]:
        model=BandpowerClassifier().fit(x[train],y[train]);rr=[rows[i] for i in test];prediction=predict_streams(model,x[test],rr)
        results[name]=score(y[test],prediction,rr);results[name].update(train_cases=len(train),test_cases=len(test))
        np.save(folder/(name+'_baseline_predictions.npy'),prediction,allow_pickle=False)
        print(name,results[name]['balanced_accuracy_cell_mean'],flush=True)
    dump(folder/'baseline-results.json',results);dump(ROOT/'docs/competition-track2-baseline-results.json',results)
    return results


def audit(output=DEFAULT_OUTPUT,cache=DEFAULT_CACHE):
    """Verify published artifacts and case alignment; refresh portable sidecars."""
    folder=Path(output)/'prepared';cache=Path(cache)
    artifacts=json.loads((folder/'artifacts.json').read_text(encoding='utf-8'))
    for name,expected in artifacts.items():
        path=safe_target(folder,name)
        if path.stat().st_size!=expected['bytes'] or sha(path)!=expected['sha256']:raise ValueError('Prepared artifact changed '+name)
    x=np.load(folder/'X.npy',mmap_mode='r');y=np.load(folder/'y.npy');rows=table(folder/'cases.tsv')
    if len(rows)!=len(x) or len(y)!=len(x):raise ValueError('Case alignment length mismatch')
    for i,row in enumerate(rows):
        if int(row['case_index'])!=i or int(row['label'])!=y[i] or hashlib.sha256(x[i].tobytes()).hexdigest()!=row['signal_sha256']:raise ValueError('Serialised case/signal alignment mismatch')
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    for recording in metadata['recordings']:
        prefix=str(cache/recording['source_file'])[:-7];channels=table(prefix+'channels.tsv')
        if [c['name'] for c in channels]!=metadata['channel_names']:raise ValueError('Recording sensor alignment mismatch')
        recording['native_bids_channels']=channels
    metadata['validation']['serialised_case_metadata_verified']=True
    splits=json.loads((folder/'splits.json').read_text(encoding='utf-8'))
    reports={}
    for protocol,tr,te in [('subject_holdout',sorted(splits['train']+splits['val']),splits['test']),('cross_session',splits['cross_session_train'],splits['cross_session_test'])]:
        train_people={rows[i]['subject'] for i in tr};test_people={rows[i]['subject'] for i in te}
        train_files={rows[i]['source_file'] for i in tr};test_files={rows[i]['source_file'] for i in te}
        if set(tr)&set(te) or train_files&test_files:raise ValueError('Cases or recordings cross partitions')
        if protocol=='subject_holdout' and train_people&test_people:raise ValueError('Unintended subject overlap')
        reports[protocol]=dict(train_cases=len(tr),test_cases=len(te),train_subjects=sorted(train_people),test_subjects=sorted(test_people),shared_subjects=sorted(train_people&test_people),shared_recordings=[],shared_cases=[],subject_overlap_intentional=protocol=='cross_session')
    metadata['validation']['protocols']=reports
    # Real aeon fit/predict smoke, deliberately separate from generalisation scores.
    from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
    tr=splits['train'];te=splits['test']
    small_train=[next(i for i in tr if y[i]==c) for c in range(3)]
    small_test=[next(i for i in te if y[i]==c) for c in range(3)]
    classifier=KNeighborsTimeSeriesClassifier(distance='euclidean',n_neighbors=1)
    classifier.fit(x[small_train],y[small_train]);prediction=classifier.predict(x[small_test])
    if prediction.shape!=(3,):raise ValueError('aeon classifier compatibility failure')
    metadata['validation']['aeon_classifier_fit_predict_smoke']=dict(train_cases=3,test_cases=3,passed=True)
    dump(folder/'metadata.json',metadata)
    code=folder/'conversion_code';code.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__),code/Path(__file__).name)
    shutil.copy2(ROOT/'tools/requirements-competition-track2.txt',code/'requirements.txt')
    shutil.copy2(ROOT/'docs/competition-track2.md',folder/'README.md')
    dump(folder/'artifacts.json',{p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.rglob('*') if p.is_file() and p.name!='artifacts.json'})
    dump(ROOT/'docs/competition-track2-preparation-result.json',dict(path=str(folder),**metadata['validation']))
    print('Full source/case alignment, artifact and aeon classifier audit passed',flush=True)
    return metadata['validation']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['download','prepare','baseline','audit']);parser.add_argument('--pilot',action='store_true');parser.add_argument('--cache',type=Path,default=DEFAULT_CACHE);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args()
    if args.action=='download':download(args.cache,args.pilot)
    elif args.action=='prepare':prepare(args.cache,args.output,args.pilot)
    elif args.action=='baseline':baseline(args.output)
    else:audit(args.output,args.cache)
