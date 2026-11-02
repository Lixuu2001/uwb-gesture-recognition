"""Verify or materialize only pinned public sources; never opens UWB data."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

BASE=Path(__file__).resolve().parent

def main(materialize):
    index=json.loads((BASE/'SOURCE_INDEX.json').read_text())
    for item in index['sources']:
        relative=Path(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        target=BASE/relative
        if not target.exists():
            assert materialize, 'Missing pinned source: '+str(relative)
            assert item['url'].startswith('https://raw.githubusercontent.com/')
            data=urllib.request.urlopen(item['url'],timeout=40).read()
            assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        data=target.read_bytes()
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],str(relative)
    print(json.dumps(dict(state='pinned_public_sources_verified',sources=len(index['sources']),
                          real_data_or_labels_read=False,training_started=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--materialize',action='store_true')
    args=parser.parse_args();main(args.materialize)
