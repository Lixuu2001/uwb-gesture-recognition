from pathlib import Path
import os,json
from common import B,complete,read,write,sha,load_inputs
from stream_archive import verify
from backup import SCOPE
def main():
 c=complete();p=B/'complete_backup';m=read(p/'MANIFEST.json');v=read(p/'READBACK_VERIFIED.json');r=read(p/'DRIVE_BACKUP_RECEIPT.json');assert m['scope']==v['scope']==r['scope']==SCOPE and r['complete_modelset_verified'] and r['numeric_audit_certified'];assert m['archive_sha256']==v['archive_sha256']==r['archive_sha256'];verify(m,read(p/'READBACK_INPUT.json'))
 bound={x['path']:x['sha256'] for x in read(p/'MEMBERS.json')['members']}
 for n in ['run01/best_model.pt','run01/final_full.pt','run01/validation_unscored.npz','run01/diagnostic_unscored.npz','study/COMPLETE_AUDIT.json']:assert bound[n]==sha(B/n.removeprefix('study/'))
 assert bound['official/weights/resnet18-f37072fd.pth']==c['weights_sha256']
 assert not (B/'run01/SUMMARY.json').exists();fd=os.open(B/'FINALIZE_ONCE.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
 from engine import np,macro
 ids,users,loc,maps,_,_=load_inputs(c,False);labels=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');scores={}
 for role,support in [('validation',range(10)),('diagnostic',range(12))]:
  with np.load(B/'run01'/(role+'_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[loc[role]]);np.testing.assert_array_equal(z['predictions'],z['scores'].argmax(1));scores[role]=macro(np.array(labels[z['ids']],copy=True),z['predictions'],support)
 delta=scores['diagnostic']-c['reference_held1_macro'];passed=delta+c['tolerance']>=c['minimum_gain'];result=dict(state='complete_exploratory_full_alllayer_ResNet18_SGDM40_B15',scores=scores,delta_macro=delta,predeclared_gate_passed=passed,route='separately_register_source6_confirmation' if passed else c['fail_route'],complete_private_modelset_verified=True,source6_confirmation_complete=False,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False)
 write(B/'run01/SUMMARY.json',result);print(json.dumps(result))
if __name__=='__main__':main()
