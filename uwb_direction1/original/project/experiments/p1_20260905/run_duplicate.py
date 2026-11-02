import sys
sys.path.insert(0, str(__file__).rsplit('\\', 1)[0])
from parallel_core import run_stage

run_stage('sv_R_duplicate', 'duplicate', [2], workers=2)
