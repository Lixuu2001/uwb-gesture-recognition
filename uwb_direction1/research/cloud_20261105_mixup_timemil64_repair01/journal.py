"""Two atomic state slots; committed prefix and uncommitted suffix are preserved."""
from pathlib import Path
import json,hashlib,os,time
from common import read,write,sha
import engine as e
class StepLog:
    def __init__(self,path,updates=0,prefix_bytes=0,prefix_sha=None):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        raw=self.path.read_bytes() if self.path.exists() else b''
        assert len(raw)>=prefix_bytes
        prefix=raw[:prefix_bytes];assert len(prefix.splitlines())==updates
        if prefix_sha is not None:assert hashlib.sha256(prefix).hexdigest()==prefix_sha
        self.rows=raw.splitlines(keepends=True);assert all(x.endswith(b'\n') for x in self.rows)
        assert all(json.loads(x)['update']==i+1 for i,x in enumerate(self.rows))
        self.index=updates;self.size=prefix_bytes;self.h=hashlib.sha256(prefix);self.original=len(self.rows)
        self.f=self.path.open('ab')
    def append(self,row):
        assert row['update']==self.index+1
        if self.index<len(self.rows):
            line=self.rows[self.index];assert json.loads(line)==row,'Uncommitted suffix replay mismatch'
        else:
            line=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
            self.f.write(line);self.f.flush()
        self.h.update(line);self.size+=len(line);self.index+=1
    def prefix(self):
        self.f.flush();os.fsync(self.f.fileno())
        return dict(updates=self.index,prefix_bytes=self.size,prefix_sha256=self.h.hexdigest())
    def close(self):self.f.close()
def put_json_match(path,value):
    if Path(path).exists():assert read(path)==value,'Orphan pure epoch record changed'
    else:write(path,value)
def commit(out,state,record,prefix,reference=None):
    out=Path(out);out.mkdir(exist_ok=True)
    old=read(out/'PROGRESS.json') if (out/'PROGRESS.json').exists() else None
    epoch=state['epoch'];assert epoch==(old['epoch']+1 if old else 0)
    assert state['updates']==prefix['updates']==record['updates'] and e.semantic(state)==record['state_sha256']
    put_json_match(out/f'epoch{epoch:03}.json',record)
    slot=f'slot{epoch%2}.pt';e.atomic_save(out/slot,state)
    j=dict(epoch=epoch,updates=state['updates'],state_sha256=e.semantic(state),checkpoint_sha256=sha(out/slot),
        record_sha256=sha(out/f'epoch{epoch:03}.json'),previous_journal_sha256=None if old is None else old['journal_sha256'],
        producer_record_sha256=reference,**{k:v for k,v in prefix.items() if k!='updates'})
    path=out/f'journal{epoch:03}.json'
    if path.exists():
        d=out/'orphans';d.mkdir(exist_ok=True);path.rename(d/(path.name+'.'+str(time.time_ns())))
    write(path,j)
    write(out/'PROGRESS.json',dict(epoch=epoch,updates=state['updates'],slot=slot,checkpoint_sha256=sha(out/slot),journal_sha256=sha(path),**{k:v for k,v in prefix.items() if k!='updates'}))
def load(out):
    out=Path(out)
    if not (out/'PROGRESS.json').exists():return None,None
    p=read(out/'PROGRESS.json');previous=None
    for epoch in range(p['epoch']+1):
        j=read(out/f'journal{epoch:03}.json');r=read(out/f'epoch{epoch:03}.json')
        assert j['epoch']==r['epoch']==epoch and j['previous_journal_sha256']==previous
        assert j['record_sha256']==sha(out/f'epoch{epoch:03}.json') and j['state_sha256']==r['state_sha256']
        previous=sha(out/f'journal{epoch:03}.json')
    assert previous==p['journal_sha256'] and p['slot']==f'slot{p["epoch"]%2}.pt'
    assert sha(out/p['slot'])==p['checkpoint_sha256']==j['checkpoint_sha256']
    state=e.torch.load(out/p['slot'],map_location='cpu',weights_only=True)
    assert e.semantic(state)==j['state_sha256'] and state['epoch']==p['epoch'] and state['updates']==p['updates']
    return state,p
