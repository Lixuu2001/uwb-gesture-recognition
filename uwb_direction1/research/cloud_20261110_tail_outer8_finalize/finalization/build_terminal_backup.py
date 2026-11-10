"""Private final score/source/ID audit plus nested existing full8 model recovery."""
from common import *
def main():
 resource_check();sys.path.insert(0,str(ROOT));from stream_archive import build,verify
 files={}
 for f in HERE.rglob('*'):
  if f.is_file() and f.suffix in ['.py','.json','.npz','.md'] and '__pycache__' not in f.parts:files['finalization/'+str(f.relative_to(HERE))]=f
 for n in ['SUMMARY.json','FINAL_RESULT_STATE.json','FINALIZE_ONCE.lock','FROZEN_HASHES.json','PROTOCOL.json','METHOD_SELECTION.json','PREPARATION_CHAIN.json','PRODUCTION_RELEASE.json','PHASE_LEDGER.json','EXTERNAL_PHASE_CHARGES.json']:files['phase/'+n]=ROOT/n
 for n in ['HELD8_COMPLETE_PRIVATE_INDEX.json','ALL8_PRIVATE_COMPLETE_SCORE_READY.json','HELD8_COMPLETE_AUDIT.json']:files['recovery/'+n]=ROOT/'observations'/n
 m=build(ROOT/'terminal_backup','tail_outer8_once_result_and_recovery_20261110',files,'complete once-scored full8 finalization source/results/private IDs+truth audit and nested all8 existing model recovery index; models reside in their144 privately verified prior chunks',limit=32*1024**2)
 rows=[dict(**c,path=str(ROOT/'terminal_backup'/c['name'])) for c in m['chunks']];write(ROOT/'terminal_backup/LOCAL_VERIFIED.json',verify(m,rows));print(json.dumps(dict(parts=len(rows),payload=m['tracked_payload_files'],bytes=m['archive_bytes'],sha256=m['archive_sha256'])))
if __name__=='__main__':main()
