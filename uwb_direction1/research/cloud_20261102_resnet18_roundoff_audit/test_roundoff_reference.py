from pathlib import Path
import json,unittest
from roundoff_reference import np,pixel_error_bound,verify_pixels
class Test(unittest.TestCase):
 def test_analytic_bound_and_stress_images(self):
  rng=np.random.default_rng(20261029);x=rng.normal(size=(8,3,90,189)).astype(np.float32);x[0]=1;x[1]=0;x[2]*=np.float32(1e30);x[3]*=np.float32(1e-30);x[4]=np.indices((90,189)).sum(0)%2;x[5]=0;x[5,:,45,94]=1;x[6,:,:45]=0;x[6,:,45:]=1;x[7]=np.linspace(-1,1,90*189).reshape(90,189)
  maximum,count=verify_pixels(x);self.assertLess(maximum,pixel_error_bound());self.assertEqual(count,8*3*224*224)
  (Path(__file__).resolve().parent/'SYNTHETIC_ROUNDOFF_CHECK.json').write_text(json.dumps(dict(state='passed_label_free_stress_images_under_analytic_IEEE754_bound',dtype='float32',unit_roundoff=2**-24,bound=pixel_error_bound(),maximum_synthetic_error=maximum,pixels=count,patterns=['constant','zero','random','1e30','1e-30','checkerboard','impulse','step','ramp'],bound_derived_from_dtype_dimensions_not_observed_UWB_error=True,no_UWB_values_or_labels_loaded=True,scientific_pipeline_changed=False),indent=2)+'\n')
if __name__=='__main__':unittest.main(verbosity=2)
