"""Continue reviewed diagnosis downloads, serialising DSP after competition LOSO.

The first full downloader must already be running or completed. This controller
does not register datasets. Mixed-rate pilots must pass before the second bulk
download. All native transfers retain the existing pinned source/hash checks.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.openeegbench/eegdash-diagnosis'
COMPETITION = Path('D:/Data/EEGCompetitions/NeuralInterfaces2026Track2')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run():
    from tools.competition_track2 import dump
    import psutil
    state_path = LOCAL / 'download-continuation.json'

    def state(status, **extra):
        dump(state_path, dict(status=status, pid=os.getpid(), updated=time.time(), **extra))

    def wait_receipt(path, description, live_pid=False):
        while True:
            receipt = read(path)
            if receipt['status'] == 'complete':
                return receipt
            if receipt['status'] in ['error', 'failed', 'stopped_for_review']:
                raise RuntimeError(description + ': ' + receipt.get('error', receipt['status']))
            if live_pid and not psutil.pid_exists(receipt['pid']):
                raise RuntimeError(description + ': worker exited without completion receipt')
            time.sleep(15)

    def worker(key, action):
        subprocess.run([sys.executable, '-u', str(LOCAL / 'run.py'), key, action], check=True)
        receipt = read(LOCAL / (key + '-' + action + '.json'))
        if receipt['status'] != 'complete':
            raise RuntimeError('Missing successful receipt for ' + key + '/' + action)
        return receipt

    try:
        state('running', stage='first_cohort_full_download')
        wait_receipt(LOCAL / 'psychosis1-download-full.json', 'First cohort download', True)
        state('running', stage='waiting_for_loso_before_mixed_rate_pilots')
        wait_receipt(COMPETITION / 'selected-loso/state.json', 'LOSO evaluation')
        for key in ['psychosis1', 'psychosis2']:
            state('running', stage=key + '_mixed_rate_pilot')
            receipt = worker(key, 'pilot')
            result = receipt['result']
            if not result['validation']['channel_order_verified_all_recordings']:
                raise RuntimeError('Channel-order verification missing')
            if not result['validation']['aeon']['load_fit_predict']:
                raise RuntimeError('aeon pilot check failed')
        lock = read('D:/Data/EEGSources/EEGDash/ds003947/original-source-lock.json')
        required = sum(f['bytes'] for f in lock['native_files'])
        free = shutil.disk_usage('D:/').free
        if free < required + 30_000_000_000:
            raise RuntimeError('Insufficient space for second source plus 30GB conversion reserve')
        state('running', stage='second_cohort_full_download')
        worker('psychosis2', 'download-full')
        state('downloads_complete', next='Review full-source metadata, recording overlap and conversion storage footprint before full conversion/admission')
    except BaseException as exc:
        state('stopped_for_review', error=repr(exc))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    run()
