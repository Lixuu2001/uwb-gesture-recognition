"""Initial models were independently seeded, only user metadata/IDs read here."""
from common import B,config,read,write,sha
from pathlib import Path
import numpy as np,shutil
from roles_gate import roles
def main():
    c,p=config();assert sha(c['ids'])==c['ids_sha256']
    with np.load(c['ids']) as z:ids=z['ids'];users=z['users']
    d=B/'assets';d.mkdir(exist_ok=False);shutil.copyfile(c['ids'],d/'source_ids.npz')
    for held in range(1,7):
        loc=roles(ids,users,held);np.savez_compressed(d/f'held{held}_roles.npz',**{k:ids[v] for k,v in loc.items()})
    seeds={arm:read(B/'seed'/(arm+'.json')) for arm in ['timemil','shuffle']}
    for arm,r in seeds.items():assert sha(B/r['path'])==r['sha256']
    write(B/'SEED_FILES.json',seeds)
    write(B/'ASSET_FILES.json',{str(f.relative_to(B)):sha(f) for f in d.iterdir()})
    write(B/'INITIAL_ASSET_CHECK.json',dict(state='initial_seed_models_and_all6_disjoint_role_IDs_ready',new_UWB_fit=False,labels_decoded=False,actual_IDs_private_only=True))
if __name__=='__main__':main()
