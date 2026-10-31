"""Keep native-checkpoint inspection observational; no model/RNG checks relaxed."""
from contextlib import contextmanager
import random,sys
from pathlib import Path
from ops import snapshot,equal

@contextmanager
def isolate_inspection_python_rng(holder,model,epoch,total,batch,identity):
 before=snapshot(holder,model,epoch,total,batch,identity)
 try:yield
 finally:
  # Constructors used only to inspect separate native checkpoints consume
  # Python random state. They are absent from the producer inference path.
  # Restore only that observational stream, then demand every state be exact.
  random.setstate(before['native']['python_rng'])
  equal(before,snapshot(holder,model,epoch,total,batch,identity))
