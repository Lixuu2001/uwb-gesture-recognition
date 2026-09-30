"""Actual transfer round trips; fixtures are not UWB method experiments."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / 'tools/transfer.py'
spec = importlib.util.spec_from_file_location('uwb_transfer', TOOL)
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.code = self.root / 'worktree'
        self.data = self.root / 'data'
        self.package = self.root / 'package'
        self.code.mkdir()
        self.data.mkdir()
        (self.code / 'model.py').write_bytes(b'print("worktree")\r\n')
        (self.data / 'model.py').write_bytes(b'print("old")\n')
        for name in transfer.RAW_FILES:
            p = self.data / name
            p.parent.mkdir(parents=True, exist_ok=True)
            content = bytes(range(256)) * (8500 if name.endswith('/X_clutterremoved.npy') else 2)
            p.write_bytes(content)
        self.args = argparse.Namespace(code_root=self.code, data_root=self.data, out=self.package,
                                       code_only=False, include_prepared=False, no_features=False, chunk_mib=1)

    def pack(self):
        with contextlib.redirect_stdout(io.StringIO()):
            transfer.pack(self.args)
        return transfer.load_manifest(self.package)

    def test_round_trip_multiple_chunks_is_byte_identical(self):
        original = {name: (self.data / name).read_bytes() for name in transfer.RAW_FILES}
        manifest = self.pack()
        x = next(e for e in manifest['files'] if e['path'] == transfer.RAW_FILES[0])
        self.assertEqual(len(x['chunks']), 3)
        restored = self.root / 'restored'
        with contextlib.redirect_stdout(io.StringIO()):
            transfer.restore(argparse.Namespace(package=self.package, out=restored))
        for name, content in original.items():
            self.assertEqual((restored / name).read_bytes(), content)
            self.assertEqual((self.data / name).read_bytes(), content)
        self.assertEqual((self.package / 'code/project/model.py').read_bytes(),
                         (self.code / 'model.py').read_bytes())
        self.assertEqual(json.loads((restored / 'RESTORE_VERIFIED.json').read_text())['state'], 'verified')

    def test_corrupt_chunk_fails_before_creating_restore_directory(self):
        manifest = self.pack()
        chunk = self.package / manifest['files'][0]['chunks'][0]['path']
        content = bytearray(chunk.read_bytes())
        content[-1] ^= 1
        chunk.write_bytes(content)
        restored = self.root / 'restored'
        with self.assertRaisesRegex(ValueError, 'Corrupt chunk'):
            transfer.restore(argparse.Namespace(package=self.package, out=restored))
        self.assertFalse(restored.exists())

    def test_missing_input_does_not_create_export(self):
        (self.data / transfer.RAW_FILES[1]).unlink()
        with self.assertRaisesRegex(ValueError, 'Required historical inputs'):
            self.pack()
        self.assertFalse(self.package.exists())

    def test_loader_lineage_transfers_but_checkpoints_do_not(self):
        old = self.data / 'experiments/p1_20260905/cross_user_meta/runs_s2026'
        (old / 'matrices').mkdir(parents=True)
        (old / 'manifest.json').write_text('{"config":{"seed":2026}}', encoding='utf-8')
        (old / 'matrices/fold1.npz').write_bytes(b'frozen-source-matrix')
        recent = self.data / 'experiments/p1_20260928/baseline_s2026/fold1/warmup'
        recent.mkdir(parents=True)
        (recent / 'features.json').write_text('{"sha256":"fixture"}', encoding='utf-8')
        (recent / 'best.pt').write_bytes(b'not-part-of-transfer')
        manifest = self.pack()
        paths = {e['path'] for e in manifest['files']}
        for p in (old / 'manifest.json', old / 'matrices/fold1.npz', recent / 'features.json'):
            self.assertIn(p.relative_to(self.data).as_posix(), paths)
        self.assertNotIn((recent / 'best.pt').relative_to(self.data).as_posix(), paths)
        self.assertEqual((self.package / 'code/.gitattributes').read_text(), '** -text -whitespace\n')

    def test_missing_chunk_fails_before_restore(self):
        manifest = self.pack()
        (self.package / manifest['files'][0]['chunks'][0]['path']).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing chunk'):
            transfer.restore(argparse.Namespace(package=self.package, out=self.root / 'restored'))
        self.assertFalse((self.root / 'restored').exists())

    def test_code_only_and_excluded_inputs(self):
        secret = self.code / '.env'
        secret.write_text('not-exported', encoding='utf-8')
        env_dir = self.code / '.venv'
        env_dir.mkdir()
        (env_dir / 'dependency.py').write_text('not-exported', encoding='utf-8')
        self.args.code_only = True
        (self.data / transfer.RAW_FILES[0]).unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            transfer.pack(self.args)
        manifest = json.loads((self.package / 'TRANSFER_MANIFEST.json').read_text())
        self.assertEqual(manifest['state'], 'code_only')
        self.assertFalse((self.package / 'chunks').exists())
        self.assertFalse((self.package / 'code/project/.env').exists())
        self.assertFalse((self.package / 'code/project/.venv').exists())

    def test_path_traversal_and_windows_paths_rejected(self):
        self.pack()
        path = self.package / 'TRANSFER_MANIFEST.json'
        original = json.loads(path.read_text())
        for bad in ('../escape', '/absolute', 'C:/windows', 'nested\\windows', 'a/../../escape'):
            original['files'][0]['path'] = bad
            path.write_text(json.dumps(original), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Invalid portable'):
                transfer.load_manifest(self.package)

    def test_duplicate_chunks_rejected(self):
        manifest = self.pack()
        manifest['files'].append(manifest['files'][0])
        transfer.write_json(self.package / 'TRANSFER_MANIFEST.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Duplicate restored'):
            transfer.load_manifest(self.package)

    def test_original_hash_checked_after_decompression(self):
        manifest = self.pack()
        manifest['files'][0]['sha256'] = '0' * 64
        transfer.write_json(self.package / 'TRANSFER_MANIFEST.json', manifest)
        restored = self.root / 'restored'
        with self.assertRaisesRegex(ValueError, 'Original-file hash'):
            transfer.restore(argparse.Namespace(package=self.package, out=restored))
        self.assertFalse((restored / 'RESTORE_VERIFIED.json').exists())

    def test_refuse_overwrite_and_nested_outputs(self):
        self.pack()
        with self.assertRaisesRegex(ValueError, 'new empty output'):
            self.pack()
        self.args.out = self.data / 'export'
        with self.assertRaisesRegex(ValueError, 'outside both'):
            self.pack()
        restored = self.root / 'occupied'
        restored.mkdir()
        (restored / 'keep.txt').write_text('keep', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'new empty directory'):
            transfer.restore(argparse.Namespace(package=self.package, out=restored))
        self.assertEqual((restored / 'keep.txt').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
