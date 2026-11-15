"""Transfer and submit the authorised isolated HC2 experiment to Tony's Iridis6 space.

All networking starts on the desktop. Remote operations are local Git worktree
creation, own-file writes and own-job submission; no package/network operations.
"""
from pathlib import Path
import json
import shlex
import subprocess

REMOTE = r'''
import json,pathlib,shutil,subprocess,sys,tarfile
home=pathlib.Path.home()
root=home/'Competitions/NeuralInterfaces2026Track2/HC2'
repo=home/'Code/tsml-eval-proteus-hc2-20261110'
base=home/'Code/tsml-eval'
commit='ce0d1131dca7c2f432ae005162284acfe28ce4be'
if root.exists() or repo.exists():raise RuntimeError('Target exists: inspect rather than overwrite')
actual=subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD']).decode().strip()
if actual!=commit:raise RuntimeError('Base checkout changed; review source before deploying')
root.mkdir(parents=True)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
 for member in archive:
  target=(root/member.name).resolve()
  if root.resolve() not in target.parents or not member.isfile():raise RuntimeError('Unsafe payload path/type')
  target.parent.mkdir(parents=True,exist_ok=True)
  with archive.extractfile(member) as source,target.open('xb') as dest:shutil.copyfileobj(source,dest)
subprocess.run(['git','-c','core.hooksPath=/dev/null','-C',str(base),'worktree','add','--detach',str(repo),commit],check=True)
for source in (root/'overlay').rglob('*'):
 if source.is_file():
  dest=repo/source.relative_to(root/'overlay')
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
submitted=subprocess.check_output(['sbatch','--parsable',str(root/'preflight.sbatch')]).decode().strip().split(';')[0]
receipt=dict(status='preflight_submitted',job_id=submitted,stage=str(root),repo=str(repo),base_commit=commit,remote_external_calls=False)
(root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
'''


def main():
    stage=Path('D:/Data/EEGCompetitions/NeuralInterfaces2026Track2/hc2-cluster-stage')
    ssh=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','BatchMode=yes',
         '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=15',
         'ajb2u23@iridis6_b.soton.ac.uk','python3 -c '+shlex.quote(REMOTE)]
    with (stage/'payload.tar').open('rb') as payload:
        result=subprocess.run(ssh,stdin=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    print(result.stdout,flush=True)
    if result.stderr:print(result.stderr,flush=True)
    if result.returncode:raise RuntimeError('Remote deployment failed; inspect before retry')
    receipt=json.loads(result.stdout.strip().splitlines()[-1])
    path=Path(__file__).resolve().parents[1]/'docs/competition-track2-hc2-submission.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
