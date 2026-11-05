from pathlib import Path
import json,hashlib
B=Path(__file__).resolve().parent
def protocol():return json.loads((B/'PROBE_CONFIG.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ready():
    c=protocol()
    for n,h in c['native_dependencies'].items():assert sha(n)==h,n
    for n,h in c['author_files'].items():assert sha(n)==h,n
    for n,h in json.loads((B/'PROBE_HASHES.json').read_text()).items():assert sha(B/n)==h,n
    return c
