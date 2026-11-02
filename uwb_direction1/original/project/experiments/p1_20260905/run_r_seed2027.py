"""Confirmatory seed for the pre-registered R amplitude+phase gate."""
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 1)[0])
from parallel_core import run_stage

# Run the matched R amplitude baseline first, then the phase+amplitude arm.
# Both use seed 2027 so their paired comparison is meaningful.
run_stage('sv_R_off', 'off', [2], seed=2027, workers=2)
run_stage('sv_R_concat', 'concat', [2], seed=2027, workers=2)
