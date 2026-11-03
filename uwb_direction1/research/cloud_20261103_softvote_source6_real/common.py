from pathlib import Path
import hashlib,json,os,sys,time
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(4194304),b''):h.update(x)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    if tmp.exists():
        orphan=p.parent/'orphans';orphan.mkdir(exist_ok=True)
        tmp.rename(orphan/(tmp.name+'.'+str(time.time_ns())))
    with tmp.open('x') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def config():
    c=read(B/'CONFIG.json');p=read(B/'METHOD_LOCK.json')
    assert sha(B/'METHOD_LOCK.json')==c['method_lock_sha256']
    assert p['seed']==20261029 and p['weights']==[.5,.5] and p['temperature']==1
    assert p['new_held_users']==[2,3,4,5,6] and p['users7_8_labels_forbidden']
    return c,p
def ready():
    assert not sys.flags.optimize
    c,p=config();m=read(B/'REMOTE_REAL_PREREGISTRATION.json')
    assert m['state']=='all_real_files_exact_remote_readback_verified' and m['branch_head_verified']
    assert m['frozen_sha256']==sha(B/'FROZEN_HASHES.json')
    for n,h in read(B/'FROZEN_HASHES.json').items():assert sha(B/n)==h,n
    for n,h in read(B/'DEPENDENCIES.json')['files'].items():assert sha(n)==h,n
    assert sha(c['prepared_coupled_result'])==c['prepared_coupled_result_sha256']
    assert not read(c['prepared_coupled_result'])['new_UWB_fit_started']
    pilot=Path(p['reused_held1_root']);assert sha(pilot/'SUMMARY.json')==p['reused_summary_sha256']
    s=read(pilot/'SUMMARY.json');assert s['predeclared_diagnostic_gate_passed'] and not s['users7_8_labels_used']
    assert sha(pilot/'private_backup/DRIVE_BACKUP_RECEIPT.json')==p['reused_private_receipt_sha256']
    for arm,row in read(B/'SEED_FILES.json').items():assert sha(B/row['path'])==row['sha256'],arm
    for n,h in read(B/'ASSET_FILES.json').items():assert sha(B/n)==h,n
    assert sha(c['ids'])==c['ids_sha256'] and sha(c['maps'])==c['maps_sha256']
    for n,h in c['input_sha256'].items():assert sha(Path(c['data'])/'cache'/(n+'.npy'))==h,n
    for files in c['baselines'].values():
        for row in files.values():assert sha(row['path'])==row['sha256']
    for row in c['held1_member_summaries'].values():assert sha(row['path'])==row['sha256']
    capacity=read(B/'CAPACITY_VERIFIED.json')
    assert capacity['resource_gate_passed'] and not capacity['new_UWB_fit_started']
    assert capacity['projected_perfold_wall_seconds']<=p['per_fold_wall_seconds']
    assert capacity['projected_perfold_CPU_seconds']<=p['per_fold_CPU_seconds']
    return c,p
def require_initial():
    c,p=ready();r=read(B/'INITIAL_PRIVATE_INSURANCE.json')
    assert r['state']=='initial_source6_models_roles_sources_private_raw_allmember_verified'
    assert r['frozen_sha256']==sha(B/'FROZEN_HASHES.json') and r['before_new_UWB_fit'] is True
    m=read(B/'initial_backup/MANIFEST.json');v=read(B/'initial_backup/READBACK_VERIFIED.json')
    assert m['scope']==v['scope']==c['initial_archive_scope']
    assert v['state']=='remote_archive_and_all_members_verified'
    assert r['archive_sha256']==m['archive_sha256']==v['archive_sha256']
    assert sha(B/'initial_backup/MEMBERS.json')==m['members_manifest_sha256']
    from stream_archive import verify
    rows=read(B/'initial_backup/READBACK_INPUT.json');privacy(rows)
    assert verify(m,rows)==v
    return c,p
def privacy(rows):
    assert rows and len({x['file_id'] for x in rows})==len(rows)
    for x in rows:
        assert x['raw_fetch'] is True and x['include_base64'] is False and x['download_verified'] is True
        m=x['metadata'];assert m['shared'] is False and m['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
        assert m['permissions']==[dict(role='owner',type='user')] and m['size_bytes']==x['bytes']
def successful(directory):
    t=read(Path(directory)/'TERMINATION.json')
    assert t['exit_code']==0 and t['resource_stop_reason'] is None and not t['wall_timeout']
    assert t['cgroup_OOM_kill_delta']==0
    return t
def process_identity(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();parts=raw[raw.rfind(')')+2:].split()
        return dict(pid=int(pid),startticks=int(parts[19]),state=parts[0],CPU_seconds=(int(parts[11])+int(parts[12]))/os.sysconf('SC_CLK_TCK'),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:return None
def alive(identity):
    now=process_identity(identity['pid'])
    return bool(now and now['state']!='Z' and now['boot_id']==identity['boot_id'] and now['startticks']==identity['startticks'])
