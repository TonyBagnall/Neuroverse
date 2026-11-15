"""Incremental own-file SSH pull with staging, CSV checks and overwrite backups.

Reads only the specified remote tree. No scheduler or remote file changes.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time


def validate_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        header = [f.readline() for _ in range(3)]
        if not all(header):raise ValueError('Incomplete result header')
        fields = header[2].strip().split(',')
        classes = int(fields[5]);expected_accuracy = float(fields[0])
        count = correct = 0
        for row in csv.reader(f):
            if not row:continue
            if len(row) < 3 + classes:raise ValueError('Incomplete result row')
            truth, prediction = int(row[0]), int(row[1])
            probabilities = [float(v) for v in row[3:3+classes]]
            if not 0 <= truth < classes or not 0 <= prediction < classes:
                raise ValueError('Class index outside mapping')
            if not all(math.isfinite(v) and -1e-9 <= v <= 1+1e-9 for v in probabilities):
                raise ValueError('Invalid probability')
            if abs(sum(probabilities)-1) > 1e-5:raise ValueError('Probability sum differs from one')
            count += 1;correct += truth == prediction
        if not count:raise ValueError('No predictions')
        if abs(correct/count - expected_accuracy) > 1e-8:
            raise ValueError('Stored accuracy disagrees with predictions')
    return dict(cases=count, classes=classes, accuracy=correct/count)


REMOTE = r'''
import json,sys,tarfile,pathlib
request=json.load(sys.stdin);root=pathlib.Path(request['root']).expanduser().resolve()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|',format=tarfile.PAX_FORMAT) as archive:
 for entry in request['files']:
  path=(root/entry['path']).resolve()
  if root not in path.parents or not path.is_file():raise RuntimeError('Unsafe source path')
  stat=path.stat()
  if stat.st_size!=entry['bytes'] or stat.st_mtime_ns!=entry['mtime_ns']:raise RuntimeError('Source changed since inventory: '+entry['path'])
  archive.add(str(path),arcname=entry['path'],recursive=False)
'''


def pull(inventory, remote_root, local_root, report, host='ajb2u23@iridis6_b.soton.ac.uk'):
    inventory=json.loads(Path(inventory).read_bytes());local_root=Path(local_root)
    run=time.strftime('%Y%m%d-%H%M%S');staging=local_root.parent/(local_root.name+'-incoming-'+run)
    files=[];skipped_recent=[];skipped_older=[]
    for entry in inventory['files']:
        rel=Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe relative path')
        if rel.suffix!='.csv':continue  # Model checkpoints are a separate decision.
        if time.time()-entry['mtime_ns']/1e9 < 90:skipped_recent.append(entry['path']);continue
        path=local_root/rel
        if path.is_file() and int(path.stat().st_mtime)>entry['mtime_ns']//1_000_000_000:
            skipped_older.append(entry['path']);continue
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or int(path.stat().st_mtime)!=entry['mtime_ns']//1_000_000_000:
            files.append(entry)
    result=dict(source_host=host, remote_root=remote_root, local_root=str(local_root),
                inventory_files=len(inventory['files']),requested_files=len(files),
                requested_bytes=sum(f['bytes'] for f in files),skipped_recent=skipped_recent,skipped_older=skipped_older,
                fetched=[],invalid=[],status='running')
    Path(report).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not files:
        result['status']='complete';Path(report).write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    staging.mkdir(parents=True,exist_ok=False)
    wanted={f['path']:f for f in files}
    args=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
          '-o','UpdateHostKeys=no','-o','ConnectTimeout=15','-o','ServerAliveInterval=60',host,
          'python3 -c '+shlex.quote(REMOTE)]
    with (staging/'transfer.stderr').open('wb') as errors:
        p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors)
        p.stdin.write(json.dumps(dict(root=remote_root,files=files)).encode());p.stdin.close()
        try:
            with tarfile.open(fileobj=p.stdout,mode='r|') as archive:
                for member in archive:
                    if not member.isfile() or member.name not in wanted:raise ValueError('Unexpected tar member')
                    path=(staging/member.name).resolve()
                    if not path.is_relative_to(staging.resolve()):raise ValueError('Unsafe archive destination')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(member) as src,path.open('xb') as dest:shutil.copyfileobj(src,dest)
                    if path.stat().st_size!=wanted[member.name]['bytes']:raise ValueError('Truncated transfer')
                    os.utime(path,ns=(wanted[member.name]['mtime_ns'],wanted[member.name]['mtime_ns']))
            if p.wait()!=0:raise RuntimeError('SSH transfer failed: '+(staging/'transfer.stderr').read_text())
        except BaseException:
            p.kill();p.wait();raise
    backup=local_root.parent/(local_root.name+'-replaced-'+run)
    for entry in files:
        src=staging/entry['path']
        try:checks=validate_csv(src)
        except Exception as exc:
            result['invalid'].append(dict(path=entry['path'],error=str(exc)));continue
        dest=local_root/entry['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            old=backup/entry['path'];old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,old)
        digest=hashlib.sha256(src.read_bytes()).hexdigest();src.replace(dest)
        result['fetched'].append(dict(**entry,sha256=digest,**checks))
        if len(result['fetched'])%100==0:print('Merged',len(result['fetched']),'/',len(files),flush=True)
    result.update(status='complete',staging=str(staging),backup=str(backup),
                  fetched_bytes=sum(f['bytes'] for f in result['fetched']))
    Path(report).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--remote-root',required=True)
    parser.add_argument('--local-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--host',default='ajb2u23@iridis6_b.soton.ac.uk')
    args=parser.parse_args();result=pull(args.inventory,args.remote_root,args.local_root,args.report,args.host)
    summary={k:v for k,v in result.items() if k not in ['fetched','invalid','skipped_older','skipped_recent']}
    summary['skipped_older_count']=len(result.get('skipped_older',[]))
    summary['skipped_recent_count']=len(result.get('skipped_recent',[]))
    print(json.dumps(summary,indent=2))
    print('Invalid files quarantined:',len(result['invalid']))
