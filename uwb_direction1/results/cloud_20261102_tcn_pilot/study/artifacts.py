"""Atomic evidence writes and label-blind native file observation isolation."""
from pathlib import Path
import json
def write(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def save(p,obj,torch):
 p=Path(p);tmp=p.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(p)
