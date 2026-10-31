"""Recreate isolated pinned wheels; honors configured session networking.

No global packages are installed/replaced. Base Python/NumPy/Torch environment
must match RUNTIME.json; this is dependency recovery, not whole-job resume.
"""
import hashlib,json,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
 assert sys.argv[1:]==['--install']
 wheels=B/'dependency_wheels';wheels.mkdir(exist_ok=True)
 req=B/'DEPENDENCY_REQUIREMENTS.txt'
 subprocess.run([sys.executable,'-m','pip','download','--require-hashes','--no-deps','--only-binary=:all:','-r',str(req),'-d',str(wheels)],check=True)
 for r in json.loads((B/'WHEEL_HASHES.json').read_text()):
  p=wheels/r['filename'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['filename']
 assert not (B/'deps').exists(),'Refuse to overwrite an existing runtime; inspect its manifest first.'
 subprocess.run([sys.executable,'-m','pip','install','--require-hashes','--no-deps','--no-index','--find-links',str(wheels),'--target',str(B/'deps'),'-r',str(req)],check=True)
 for n,h in json.loads((B/'DEPENDENCIES.json').read_text())['files'].items():assert sha(B/'deps'/n)==h,n
 print('Isolated runtime files match pinned manifest; no UWB data or training used.')
