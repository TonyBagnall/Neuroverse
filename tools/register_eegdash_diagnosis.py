"""Register a fully validated local diagnostic preparation, preserving MI counts."""
import importlib
import json
from pathlib import Path
import re
from tools.prepare_moabb import dump

ROOT=Path(__file__).resolve().parents[1]
# Study selection excludes retired preparations plus non-scalp modalities.
EXCLUDED={'Blink', 'MatchWords', 'Sleep', 'HandMovementDirection', 'EpilepticSeizures', 'MotorImagery', 'FaceDetection', 'MindReading', 'ButtonPress'}


def register(result):
    from eegbench import datasets
    importlib.reload(datasets)
    if result['dataset'].endswith('Pilot'):raise ValueError('Pilot is not an archive admission')
    folder=Path(result['path']);name=result['dataset'];m=json.loads((folder/(name+'_metadata.json')).read_text(encoding='utf-8'))
    if result['validation']['shared_subjects'] or result['validation']['cross_split_duplicate_cases']:raise ValueError('Diagnostic archive requires participant-independent clean split')
    row=(name,result['channels'],result['classes'],result['timepoints'],result['subjects'],result['train_cases'],result['test_cases'],result['frequency'],'Med')
    existing=[r for r in datasets._TABLE if r[0]==name]
    if existing:
        if existing!=[row]:raise ValueError('Conflicting registered dimensions')
        return
    old=len(datasets.eegbench);old_local=len(datasets.local_only);old_selected=len((ROOT/'eeg.txt').read_text().splitlines());count=old+1;local=old_local+1
    p=ROOT/'eegbench/datasets.py';t=p.read_text(encoding='utf-8');marker='_TABLE.sort(key=lambda row: row[0])'
    if t.count(marker)!=1:raise ValueError('Registry marker changed')
    t=t.replace(marker,'_TABLE += '+repr([row])+'\n\n'+marker).replace(f'and {old_local} additional local preparations',f'and {local} additional local preparations').replace(f'All {old} prepared classification',f'All {count} prepared classification');p.write_text(t,encoding='utf-8');importlib.reload(datasets)
    selected=[n for n in datasets.eegbench if n not in EXCLUDED];(ROOT/'eeg.txt').write_text('\n'.join(selected)+'\n',encoding='utf-8')
    p=ROOT/'eegbench/tests/test_datasets.py';t=p.read_text(encoding='utf-8').replace(f'All {old} problems',f'All {count} problems').replace(f'len(set(eegbench)) == {old}',f'len(set(eegbench)) == {count}').replace(f'len(local_only) == {old_local}',f'len(local_only) == {local}');p.write_text(t,encoding='utf-8')
    p=ROOT/'README.md';t=p.read_text(encoding='utf-8')
    for a,b in [(f'archive of {old} EEG',f'archive of {count} EEG'),(f'of the {old} datasets',f'of the {count} datasets'),(f'The {old_local}\nnewer preparations',f'The {local}\nnewer preparations'),(f'All {old} are in',f'All {count} are in'),(f'requests all {old} and',f'requests all {count} and')]:t=t.replace(a,b)
    p.write_text(t,encoding='utf-8')
    p=ROOT/'docs/datasets.md';t=p.read_text(encoding='utf-8');t=re.sub(r'(?s)(<!-- DATASETS:START -->).*?(<!-- DATASETS:END -->)',lambda a:a[1]+'\n\n'+datasets.metadata_markdown()+'\n\n'+a[2],t);t=t.replace(f'contains {old} prepared',f'contains {count} prepared').replace(f'contains {old_selected} problems',f'contains {len(selected)} problems')
    t+=f'\n### {name}\n\n{m["source_dataset"]}. {result["subjects"]} participants; {result["train_cases"]}/{result["test_cases"]} TRAIN/TEST cases, {result["channels"]} channels x {result["timepoints"]} samples at {result["frequency"]} Hz. Classes: '+', '.join(m['label_mapping'].values())+'.\n\n'+m['source_and_conversion_notes']+'\n\n'+m['split_definition']+f' Native sensor/history metadata, case-aligned tables, exclusions and validation accompany D:/Data/EEG/{name}. See [clinical assimilation plan](eegdash-diagnosis-plan.md).\n';p.write_text(t,encoding='utf-8')
    p=ROOT/'docs/dataset-provenance.md';t=p.read_text(encoding='utf-8')+f'\n### {name}: EEGDash diagnostic addition\n\nOriginal source: [{m["source_doi"]}]({m["original_source_url"]}); aliases {m["source_dataset"]}, EEGDash `{m["eegdash_id"]}`. Data licence {m["licence"]}. {m["paper_reference"]}. Source revision: {m["source_revision"]}. {m["source_and_conversion_notes"]} See [clinical plan](eegdash-diagnosis-plan.md) and the source/manifest hashes in the prepared folder.\n';p.write_text(t,encoding='utf-8')
    p=ROOT/'docs/eegdash-diagnosis-progress.json';progress=json.loads(p.read_text());progress['completed']=list(dict.fromkeys(progress['completed']+[name]));progress.update(status='continuing_source_review',registered=count,selected=len(selected));dump(p,progress)
    p=ROOT/'docs/eegdash-diagnosis-summary.json';rows=json.loads(p.read_text()) if p.exists() else [];fields=['dataset','train_cases','test_cases','subjects','channels','timepoints','classes','frequency','size_GB','licence'];rows=[r for r in rows if r['dataset']!=name]+[{k:result[k] for k in fields}];dump(p,rows)
    p=ROOT/'docs/assimilation-progress.json';main=json.loads(p.read_text());main.update(registered=count,selected=len(selected),updated='2026-11-09');main['in_progress']['eegdash_diagnosis']=dict(target_additional=20,completed=progress['completed'],report='docs/eegdash-diagnosis-progress.json');dump(p,main)
    print('Registered',name,';',len(progress['completed']),'of20diagnostic additions;',count,'total problems',flush=True)
