"""Register explicit local CPU measurements and conservative helper charges, once."""
import sys
from core import ROOT,read,write,sha
def main():
    held=int(sys.argv[1]);fold=ROOT/'folds'/f'held{held}';path=ROOT/'EXTERNAL_PHASE_CHARGES.json';rows=read(path) if path.exists() else [];ids=[]
    for suffix,file in [('build','BACKUP_BUILD_COST.json'),('verify','BACKUP_VERIFY_COST.json')]:
        p=fold/file;r=read(p)
        if r['exit_code']!=0:raise PermissionError('Failed backup execution')
        CPU=r['user_CPU_seconds']+r['system_CPU_seconds']+r.get('parent_CPU_seconds',0.)
        id=f'held{held}-backup-{suffix}'
        if id in {x['id'] for x in rows}:raise PermissionError('Cost already registered')
        rows.append(dict(id=id,CPU_seconds=CPU,receipt=str(p),receipt_sha256=sha(p),kind='measured_waited_process_family'));ids.append(id)
    # Upload/fetch helper Python SHA/materialization commands lack isolated CPU counters.
    # Charge a declared conservative allowance instead of silently treating them as zero.
    helper=fold/'BACKUP_HELPER_CPU_ALLOWANCE.json';parts=len(read(fold/'private_backup/MANIFEST.json')['chunks'])
    value=dict(kind='budget_charge_not_measured_CPU',CPU_seconds=2.*parts+2.,parts=parts,remote_service_CPU_unmeasured_and_not_local_execution_CPU=True)
    if helper.exists():raise PermissionError('Helper allowance already registered')
    write(helper,value);id=f'held{held}-backup-helper-uppercharge';rows.append(dict(id=id,CPU_seconds=value['CPU_seconds'],receipt=str(helper),receipt_sha256=sha(helper),kind=value['kind']));ids.append(id)
    write(path,rows);write(fold/'BACKUP_CHARGE_IDS.json',ids)
    print('Backup costs registered; finalize private fold release once after raw allmember verification')
if __name__=='__main__':main()
