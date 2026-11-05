from pathlib import Path
import json,hashlib
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protocol():return read(B/'PROBE_CONFIG.json')
def ready():
 for n,h in read(B/'PROBE_HASHES.json').items():assert sha(B/n)==h,n
 for n,h in read(B/'DEPENDENCIES.json')['files'].items():assert sha(n)==h,n
 for n,r in read(B/'AUTHOR_REFERENCES.json')['files'].items():assert sha(r['path'])==r['sha256'],n
 return protocol()
