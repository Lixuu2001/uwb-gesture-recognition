#!/usr/bin/env python3
"""Export a read-only UWB snapshot and restore byte-identical data chunks.

Uses only the Python standard library; importing this module has no side effects.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

MIB = 1024 * 1024
TEXT_SUFFIXES = {'.py', '.md', '.toml', '.yaml', '.yml', '.ini', '.cfg', '.sh', '.ps1', '.m'}
EXCLUDED_DIRS = {'.git', '.venv', 'venv', 'env', 'node_modules', '__pycache__',
                 '.tools', 'cache', 'data', 'rawdata', 'datasets', 'prepared',
                 'checkpoints', 'weights'}
RAW_FILES = ('cache/X_clutterremoved.npy', 'cache/y_clutterremoved.npy',
             'cache/subj_clutterremoved.npy',
             'experiments/p1_20260905/prepared/y.npy',
             'experiments/p1_20260905/prepared/subj.npy',
             'experiments/p1_20260905/results/data_manifest.json')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * MIB), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Invalid portable relative path')
    parts = PurePosixPath(value)
    if parts.is_absolute() or '..' in parts.parts or parts.as_posix() != value:
        raise ValueError('Invalid portable relative path: ' + value)
    return Path(*parts.parts)


def inside(path, root):
    return path.resolve().is_relative_to(root.resolve())


def source_files(root):
    for parent, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS
                         and not d.startswith('.') and not (Path(parent) / d).is_symlink())
        for name in sorted(names):
            p = Path(parent) / name
            if p.is_symlink() or name.startswith('.'):
                continue
            if p.suffix.lower() in TEXT_SUFFIXES or (name.startswith('requirements') and p.suffix == '.txt'):
                if p.stat().st_size > 2 * MIB:
                    raise ValueError('Source file unexpectedly exceeds 2 MiB: ' + str(p))
                yield p


def git_identity(root):
    def run(*args):
        p = subprocess.run(['git', '-C', str(root), *args], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.stdout.strip() if p.returncode == 0 else None
    try:
        return {'commit': run('rev-parse', 'HEAD'), 'branch': run('branch', '--show-current'),
                'status': run('status', '--porcelain=v1', '--untracked-files=normal')}
    except FileNotFoundError:
        return {'commit': None, 'branch': None, 'status': 'git unavailable'}


def metadata_files(data_root):
    old = data_root / 'experiments/p1_20260905/results'
    recent = data_root / 'experiments/p1_20260928'
    candidates = []
    if old.exists():
        candidates.extend(old.glob('*.json'))
    if recent.exists():
        candidates.extend(recent.glob('*.json'))
        candidates.extend(recent.glob('*.md'))
        candidates.extend(recent.glob('*.txt'))
        report = recent / 'report_s2026_release_v2'
        if report.exists():
            candidates.extend(report.rglob('*'))
    for p in sorted(set(candidates)):
        if p.is_file() and not p.is_symlink() and inside(p, data_root):
            if p.suffix in {'.json', '.md', '.txt', '.csv', '.py'} and p.stat().st_size <= 2 * MIB:
                yield p


def data_files(root, include_prepared, include_features):
    files = {root / name for name in RAW_FILES}
    missing = sorted(str(p.relative_to(root)) for p in files if not p.is_file())
    if missing:
        raise ValueError('Required historical inputs are missing: ' + ', '.join(missing))
    for folder in (root / 'cache', root / 'experiments/p1_20260905/prepared'):
        for p in folder.glob('*.npy'):
            if p.stat().st_size <= 16 * MIB:
                files.add(p)  # Quality masks, original IDs, labels and user metadata.
            elif include_prepared and folder.name == 'prepared':
                files.add(p)
    if include_features:
        for pattern in ('baseline_s2026/fold*/warmup/features.npz',
                        'multiview_s2026/fold*/predictions.npz'):
            files.update((root / 'experiments/p1_20260928').glob(pattern))
    for p in sorted(files):
        if p.is_symlink() or not inside(p, root):
            raise ValueError('Transfer input points outside data root: ' + str(p))
        yield p


def pack(args):
    code_root, data_root, output = (p.resolve() for p in (args.code_root, args.data_root, args.out))
    if not code_root.is_dir() or not data_root.is_dir():
        raise ValueError('Both code root and data root must be existing directories')
    if not 1 <= args.chunk_mib <= 30:
        raise ValueError('chunk-mib must be between 1 and 30')
    if inside(output, code_root) or inside(output, data_root):
        raise ValueError('Output must be outside both original project directories')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory; exports are immutable')
    # Complete input discovery before writing or hashing large files.
    raw = [] if args.code_only else list(data_files(data_root, args.include_prepared, not args.no_features))
    sources = {p.relative_to(data_root).as_posix(): p for p in source_files(data_root)}
    # The experimental worktree is authoritative when both roots contain a path.
    sources.update({p.relative_to(code_root).as_posix(): p for p in source_files(code_root)})
    sources.update({p.relative_to(data_root).as_posix(): p for p in metadata_files(data_root)
                    if p.relative_to(data_root).as_posix() not in sources})
    if not sources:
        raise ValueError('No source files found; check code-root')
    output.mkdir(parents=True, exist_ok=True)
    source_entries = []
    for rel, p in sorted(sources.items()):
        content = p.read_bytes()
        target = output / 'code' / 'project' / relative_path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        source_entries.append({'path': 'project/' + rel, 'bytes': len(content),
                               'sha256': hashlib.sha256(content).hexdigest(),
                               'origin': 'code_root' if inside(p, code_root) else 'data_root'})
    identity = {'format_version': 1, 'state': 'source_snapshot_complete',
                'code_root': str(code_root), 'data_root': str(data_root),
                'git': git_identity(code_root), 'files': source_entries,
                'scope': 'Current selected source bytes, not a complete Git history or all results.'}
    write_json(output / 'code/SOURCE_MANIFEST.json', identity)
    (output / 'code/.gitignore').write_text('__pycache__/\n*.pyc\n*.npy\n*.npz\n*.pt\n*.pth\n.env*\n', encoding='utf-8')
    entries = []
    for number, p in enumerate(raw):
        rel = p.relative_to(data_root).as_posix()
        print('Packing ' + rel, flush=True)
        original_size = p.stat().st_size
        original_mtime = p.stat().st_mtime_ns
        h = hashlib.sha256()
        chunks = []
        with p.open('rb') as src:
            for part in range(1_000_000):
                block = src.read(args.chunk_mib * MIB)
                if not block:
                    break
                h.update(block)
                name = 'chunks/file{:04d}-part{:05d}.gz'.format(number, part)
                dest = output / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('wb') as stream:
                    with gzip.GzipFile(fileobj=stream, mode='wb', filename='', compresslevel=1, mtime=0) as zipped:
                        zipped.write(block)
                if dest.stat().st_size > 32 * MIB:
                    raise ValueError('Compressed part exceeds attachment limit')
                chunks.append({'path': name, 'bytes': dest.stat().st_size,
                               'raw_bytes': len(block), 'sha256': sha256(dest)})
        if p.stat().st_size != original_size or p.stat().st_mtime_ns != original_mtime:
            raise ValueError('Input changed during export: ' + rel)
        entries.append({'path': rel, 'bytes': original_size, 'sha256': h.hexdigest(), 'chunks': chunks})
    manifest = {'format_version': 1, 'state': 'code_only' if args.code_only else 'complete',
                'compression': 'gzip-independent-byte-chunks', 'chunk_mib': args.chunk_mib,
                'source_manifest_sha256': sha256(output / 'code/SOURCE_MANIFEST.json'),
                'include_prepared': args.include_prepared, 'files': entries,
                'raw_bytes_total': sum(x['bytes'] for x in entries),
                'chunk_count': sum(len(x['chunks']) for x in entries)}
    write_json(output / 'TRANSFER_MANIFEST.json', manifest)
    write_json(output / 'code/TRANSFER_SUMMARY.json', {k: v for k, v in manifest.items() if k != 'files'})
    print(json.dumps({'state': manifest['state'], 'output': str(output),
                      'source_files': len(source_entries), 'data_files': len(entries),
                      'chunks': manifest['chunk_count'], 'raw_bytes': manifest['raw_bytes_total']}, ensure_ascii=False))


def load_manifest(package):
    manifest = json.loads((package / 'TRANSFER_MANIFEST.json').read_text(encoding='utf-8-sig'))
    if manifest.get('format_version') != 1 or manifest.get('state') != 'complete':
        raise ValueError('Expected a complete v1 data transfer')
    seen_files, seen_chunks = set(), set()
    for entry in manifest['files']:
        rel = relative_path(entry['path'])
        if rel in seen_files:
            raise ValueError('Duplicate restored path')
        seen_files.add(rel)
        total = 0
        for part in entry['chunks']:
            key = relative_path(part['path'])
            if key.parts[0] != 'chunks' or key in seen_chunks:
                raise ValueError('Invalid or duplicate chunk path')
            if not 0 < part['raw_bytes'] <= 30 * MIB:
                raise ValueError('Invalid uncompressed chunk length')
            seen_chunks.add(key)
            total += part['raw_bytes']
        if total != entry['bytes'] or entry['bytes'] < 0:
            raise ValueError('Chunk sizes do not match file size')
    return manifest


def verify_chunks(package, manifest):
    for entry in manifest['files']:
        for part in entry['chunks']:
            p = package / relative_path(part['path'])
            if not inside(p, package) or not p.is_file() or p.stat().st_size != part['bytes']:
                raise ValueError('Missing chunk or wrong size: ' + part['path'])
            if sha256(p) != part['sha256']:
                raise ValueError('Corrupt chunk: ' + part['path'])


def verify(args):
    package = args.package.resolve()
    manifest = load_manifest(package)
    verify_chunks(package, manifest)
    print('All compressed chunks verified; run restore to verify original-file hashes.')


def restore(args):
    package, output = args.package.resolve(), args.out.resolve()
    if inside(output, package) or inside(package, output):
        raise ValueError('Restored data and transfer package must be separate directories')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Restore requires a new empty directory')
    manifest = load_manifest(package)
    verify_chunks(package, manifest)  # Fail before producing data if any part is missing.
    output.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        target = output / relative_path(entry['path'])
        if not inside(target, output):
            raise ValueError('Restore target outside output')
        target.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with target.open('xb') as dest:
            for part in entry['chunks']:
                p = package / relative_path(part['path'])
                with gzip.open(p, 'rb') as src:
                    block = src.read(part['raw_bytes'] + 1)
                if len(block) != part['raw_bytes']:
                    raise ValueError('Uncompressed length mismatch: ' + part['path'])
                h.update(block)
                dest.write(block)
        if h.hexdigest() != entry['sha256']:
            raise ValueError('Original-file hash mismatch: ' + entry['path'])
    write_json(output / 'RESTORE_VERIFIED.json', {'state': 'verified', 'files': len(manifest['files']),
               'transfer_manifest_sha256': sha256(package / 'TRANSFER_MANIFEST.json')})
    print('Byte-identical restore complete: ' + str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack')
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--code-only', action='store_true')
    p.add_argument('--include-prepared', action='store_true')
    p.add_argument('--no-features', action='store_true')
    p.add_argument('--chunk-mib', type=int, default=24)
    p.set_defaults(func=pack)
    p = commands.add_parser('verify')
    p.add_argument('--package', type=Path, required=True)
    p.set_defaults(func=verify)
    p = commands.add_parser('restore')
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(func=restore)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, 'Transfer failed: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
