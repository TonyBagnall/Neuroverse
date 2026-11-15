"""Resumable reviewed clinical queue; stop on failed gates, never admit pilots.

Run in the repository scientific environment. State/logs are local and ignored.
Only Iowa full admission and the newly authorised mixed-rate psychosis pilots
are automated here; further sources require adapter/source review first.
"""
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.openeegbench/eegdash-diagnosis'


def run():
    from tools.download_eegdash_diagnosis import download
    from tools.prepare_eegdash_diagnosis import prepare
    from tools.prepare_moabb import dump
    from tools.register_eegdash_diagnosis import register
    from tools.run_mi_to_30 import verify_final
    STATE.mkdir(parents=True,exist_ok=True)
    def state(status,**extra):
        dump(STATE/'reviewed-queue.json',dict(status=status,pid=os.getpid(),updated=time.time(),**extra))
    try:
        state('running',stage='waiting_for_iowa_download')
        receipt=STATE/'iowa_rest-download-full.json'
        while True:
            current=json.loads(receipt.read_text(encoding='utf-8'))
            if current['status']=='complete':break
            if current['status']=='error':raise RuntimeError(current['error'])
            # Fail if the existing downloader has exited rather than wait forever.
            import psutil
            if not psutil.pid_exists(current['pid']):raise RuntimeError('Iowa downloader exited without completion receipt')
            time.sleep(10)
        result_path=ROOT/'docs/ds004584-diagnosis-conversion-result.json'
        state('running',stage='iowa_full_conversion')
        result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else prepare('iowa_rest')
        verify_final(result);register(result)
        for key in ['psychosis1','psychosis2']:
            state('running',stage=key+'_resampling_pilot_download')
            download(key,pilot=True)
            state('running',stage=key+'_resampling_pilot_conversion')
            result=prepare(key,pilot=True);verify_final(result)
        state('pilots_complete',next='Review psychosis pilot metadata and cross-source identity before full admission; UCSD source adapter next')
    except BaseException as exc:
        state('stopped_for_review',error=str(exc));traceback.print_exc();raise


def run_ucsd():
    """Resume the separately downloaded UCSD source, serialising registration."""
    from tools.prepare_eegdash_ucsd import prepare
    from tools.prepare_moabb import dump
    from tools.register_eegdash_diagnosis import register
    from tools.run_mi_to_30 import verify_final
    def state(status,**extra):
        dump(STATE/'ucsd-queue.json',dict(status=status,pid=os.getpid(),updated=time.time(),**extra))
    try:
        state('running',stage='waiting_for_full_download')
        while True:
            d=json.loads((STATE/'ucsd-download-full.json').read_text(encoding='utf-8'))
            if d['status']=='complete':break
            if d['status']=='error':raise RuntimeError(d['error'])
            import psutil
            if not psutil.pid_exists(d['pid']):raise RuntimeError('UCSD downloader exited without completion receipt')
            time.sleep(10)
        state('running',stage='full_conversion')
        path=ROOT/'docs/ds002778-diagnosis-conversion-result.json'
        result=json.loads(path.read_text(encoding='utf-8')) if path.exists() else prepare()
        verify_final(result)
        state('running',stage='waiting_for_serial_registration')
        while True:
            q=json.loads((STATE/'reviewed-queue.json').read_text(encoding='utf-8'))
            if q['status']=='stopped_for_review':raise RuntimeError('Other clinical queue stopped; inspect before registering UCSD')
            if q.get('stage','').startswith('psychosis') or q['status']=='pilots_complete':break
            time.sleep(10)
        register(result);state('complete',result=result)
    except BaseException as exc:
        state('stopped_for_review',error=str(exc));traceback.print_exc();raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ucsd',action='store_true')
    args=parser.parse_args()
    run_ucsd() if args.ucsd else run()
