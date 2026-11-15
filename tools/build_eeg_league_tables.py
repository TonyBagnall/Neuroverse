"""Refresh archive league tables using tsml-eval's local result reader/scorers.

Use fixed common-problem blocks; report incomplete coverage separately. No fits.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def groups(names):
    rows=json.loads((ROOT/'docs/problem-taxonomy.json').read_text())['rows']
    return {'Overall':list(names),
        'MotorImagery':sorted(r['problem'] for r in rows if r['subclass']=='Motor imagery'),
        'EvokedPotentials':sorted(r['problem'] for r in rows if r['subclass'] in
            ['ERP / P300 target detection','SSVEP','Code-modulated VEP'] or r['problem']=='WordRelatedness'),
        'Clinical':sorted(r['problem'] for r in rows if r['category']=='Clinical and physiological states')}


def league(balanced,accuracy,datasets,minimum):
    subset=balanced.loc[datasets];coverage=subset.notna().sum()
    panel=coverage[coverage>=minimum].index.tolist()
    if not panel:return pd.DataFrame(),[]
    common=subset[panel].dropna().index.tolist()
    if not common:return pd.DataFrame(),[]
    block=subset.loc[common,panel];ranks=block.rank(axis=1,ascending=False,method='average')
    wins=block.eq(block.max(axis=1),axis=0).sum()
    result=pd.DataFrame(dict(classifier=panel,coverage=[int(coverage[p]) for p in panel],
        common_problems=len(common),mean_rank=ranks.mean().values,
        mean_balanced_accuracy=block.mean().values,
        mean_accuracy=accuracy.loc[common,panel].mean().values,
        wins_including_ties=wins.values))
    result=result.sort_values(['mean_rank','mean_balanced_accuracy'],ascending=[True,False]).reset_index(drop=True)
    result.insert(0,'position',np.arange(1,len(result)+1))
    return result,common


def markdown_table(frame):
    if frame.empty:return 'No common block available.\n'
    lines=['| Rank | Classifier | Coverage | Mean rank | Balanced accuracy | Accuracy | Wins |',
           '|---:|---|---:|---:|---:|---:|---:|']
    for row in frame.itertuples():
        lines.append(f'|{row.position}|{row.classifier}|{row.coverage}|{row.mean_rank:.2f}|'
                     f'{100*row.mean_balanced_accuracy:.2f}%|{100*row.mean_accuracy:.2f}%|{row.wins_including_ties}|')
    return '\n'.join(lines)+'\n'


def build(results=Path('D:/Results/EEG')):
    from eegbench.datasets import metadata
    expected=metadata()['test_size'].to_dict()
    results=Path(results);out=results/'league-tables-20261111';out.mkdir(parents=True,exist_ok=True)
    source=Path('C:/Code/tsml-eval/local/summarise_results.py')
    spec=importlib.util.spec_from_file_location('existing_tsml_collator',source)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    names=[s.strip() for s in (ROOT/'eeg.txt').read_text().splitlines() if s.strip()]
    records=[];errors=[];cache_path=out/'metric-cache.json';cache={}
    loader_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if cache_path.exists():
        previous=json.loads(cache_path.read_text())
        if previous.get('loader_sha256')==loader_sha:cache=previous['records']
    models={}
    for category in sorted(results.iterdir()):
        if not category.is_dir() or category.name.startswith('_'):continue
        for model in sorted(category.iterdir()):
            if model.is_dir() and (model/'Predictions').is_dir():models[category.name+'/'+model.name]=model
    for number,(classifier,folder) in enumerate(models.items(),1):
        for name in names:
            path=folder/'Predictions'/name/'testResample0.csv'
            if not path.is_file():continue
            stat=path.stat();key=classifier+'|'+name;fingerprint=[stat.st_size,stat.st_mtime_ns]
            if key in cache and cache[key]['fingerprint']==fingerprint:
                records.append(cache[key]);continue
            try:
                truth,prediction,probs,order=helper._read_resample_file(path)
                if not len(truth) or len(truth)!=len(prediction):raise ValueError('Incomplete prediction vectors')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    acc=helper._score_resample(truth,prediction,probs,order,1)
                    ba=helper._score_resample(truth,prediction,probs,order,2)
                if not np.isfinite([acc,ba]).all():raise ValueError('Non-finite metric')
                digest=hashlib.sha256(np.asarray(truth,dtype='<i8').tobytes()).hexdigest()
                record=dict(classifier=classifier,dataset=name,cases=len(truth),accuracy=acc,
                            balanced_accuracy=ba,true_label_sha256=digest,fingerprint=fingerprint)
                records.append(record);cache[key]=record
            except Exception as exc:errors.append(dict(classifier=classifier,dataset=name,error=repr(exc)))
        cache_path.write_text(json.dumps(dict(loader_sha256=loader_sha,records=cache)),encoding='utf-8')
        print('Collated',number,'/',len(models),classifier,flush=True)
    frame=pd.DataFrame(records)
    # The archive metadata supplies expected TEST size; among that size use the
    # modal true-label sequence to detect non-comparable order/partition results.
    rejected=[];valid=[]
    for name,sub in frame.groupby('dataset'):
        count=expected.get(name)
        sized=sub[sub['cases']==int(count)] if count is not None and pd.notna(count) else sub
        if sized.empty:
            rejected.extend(dict(**row,exclusion='No result matches registered TEST case count') for row in sub.to_dict('records'));continue
        mode=sized['true_label_sha256'].value_counts().index[0]
        for row in sub.to_dict('records'):
            if count is not None and pd.notna(count) and row['cases']!=int(count):reason='Registered TEST case-count mismatch'
            elif row['true_label_sha256']!=mode:reason='True-label row sequence differs from modal current-size results'
            else:valid.append(row);continue
            rejected.append(dict(**row,exclusion=reason))
    valid=pd.DataFrame(valid)
    accuracy=valid.pivot(index='dataset',columns='classifier',values='accuracy').reindex(names)
    balanced=valid.pivot(index='dataset',columns='classifier',values='balanced_accuracy').reindex(names)
    accuracy.to_csv(out/'accuracy.csv',index_label='dataset');balanced.to_csv(out/'balanced_accuracy.csv',index_label='dataset')
    pd.DataFrame(rejected).to_csv(out/'excluded-result-cells.csv',index=False)
    pd.DataFrame(errors).to_csv(out/'read-errors.csv',index=False)
    coverage=pd.DataFrame(dict(classifier=balanced.columns,problems=balanced.notna().sum().values,
        available_case_mean_balanced_accuracy=balanced.mean().values,available_case_mean_accuracy=accuracy.mean().values))
    coverage.to_csv(out/'coverage.csv',index=False)
    group_names=groups(names);summary={};pages=[]
    intro=['# EEG league tables: 11 November 2026','',
        'Primary metric: test balanced accuracy; accuracy is shown alongside it. '
        'Metrics are refreshed from raw testResample0 files using tsml-eval local/summarise_results.py.', '',
        'Each primary table admits classifiers with results on at least80% of that application\'s problems, '
        'then ranks every admitted classifier on the same complete common-problem block. '
        'Mean ranks use average ties; problems receive equal weight. Wins include ties. '
        'Ranks are comparable within a table, not between application tables.','',
        'All76selected scalp EEG problems define the overall scope. MotorImagery has29problems; '
        'all evoked potentials has13 (P300/ERP, SSVEP, cVEP and N400); Clinical has17clinical/physiological problems. '
        'Other tasks remain in Overall. Competition results, MEG/ECoG and excluded archive problems are outside these tables.','',
        'Case-count mismatches against the archive registry and minority true-label row sequences are excluded and listed in the CSV audit. '
        'A matching label sequence does not prove identical signals or subject splits. This is a provisional resample0 comparison, '
        'not an independent-study analysis, a significance test or a complete benchmark. '
        'Foundation-model pretraining overlap and random-encoder ablations remain explicit result identities.','']
    for group,datasets in group_names.items():
        primary,common=league(balanced,accuracy,datasets,math.ceil(.8*len(datasets)))
        primary.to_csv(out/(group+'-league.csv'),index=False)
        (out/(group+'-common-problems.txt')).write_text('\n'.join(common)+'\n',encoding='utf-8')
        broad_models=balanced.columns[balanced.notna().sum()>=50].tolist()
        broad,broad_common=league(balanced[broad_models],accuracy[broad_models],datasets,0)
        broad.to_csv(out/(group+'-broad-panel-league.csv'),index=False)
        summary[group]=dict(scope_problems=len(datasets),admission_minimum=math.ceil(.8*len(datasets)),
                           ranked_classifiers=len(primary),common_problems=common,broad_panel_classifiers=len(broad),
                           broad_panel_common_problems=broad_common,top10=primary.head(10).to_dict('records'))
        pages.extend(['## '+group,'',f'{len(primary)}classifiers; {len(common)}common problems from {len(datasets)}in scope. '
                      f'Minimum coverage: {math.ceil(.8*len(datasets))}.','',markdown_table(primary),'',
                      'Common problems: '+', '.join(common)+'.',''])
    summary.update(results_root=str(results),output=str(out),loader=str(source),loader_sha256=loader_sha,
                   excluded_cells=len(rejected),read_errors=errors,selected_problems=names,
                   method='80% application coverage admission; fixed common blocks; broad-panel appendix uses models with50overall results')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    text='\n'.join(intro+pages+['## Coverage and audit','',
        f'{len(rejected)}result cells excluded for case-count/label-order checks; {len(errors)}read errors. '
        'Full per-problem matrices, coverage, exclusions and broader50-result-panel tables are beside this report. '
        'Available-case means in coverage.csv are descriptive and should not be read as a fair league ranking.','',
        'The broad-panel appendix preserves the earlier collation\'s approximately50-result eligibility rule; '
        'it can leave a much smaller shared block after restricting to the current76problems. '
        'No missing performance was filled with chance, zero, or another classifier\'s score.',''])
    (out/'league-tables.md').write_text(text,encoding='utf-8')
    (ROOT/'docs/eeg-league-tables-20261111.md').write_text(text,encoding='utf-8')
    (ROOT/'docs/eeg-league-tables-20261111.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    build()
