import sys
sys.path.insert(0, str(__file__).rsplit('\\', 1)[0])
from parallel_core import run_stage

run_stage('sv_R_concat_kd', 'concat', [2], workers=2,
          teacher='teacher_mean,teacher_concat')
