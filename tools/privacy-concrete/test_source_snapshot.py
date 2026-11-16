"""Actual source mutation during client materialization cannot reach private execution."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from source_snapshot import pinned_source


class SourceSnapshotTests(unittest.TestCase):
    def test_stable_verified_source_materializes_without_touching_private_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'public-model';path.write_bytes(b'original')
            with pinned_source(directory,{'public-model':hashlib.sha256(b'original').hexdigest()}):
                value=path.read_bytes()
            self.assertEqual(value,b'original')

    def test_same_length_rewrite_replacement_and_late_restore_reject_before_next_stage(self):
        for mutation in ['rewrite','replace','restore']:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'public-model';path.write_bytes(b'original');reached=[]
                with self.assertRaises(ValueError):
                    with pinned_source(directory,{'public-model':hashlib.sha256(b'original').hexdigest()}):
                        if mutation=='replace':
                            other=Path(directory)/'replacement';other.write_bytes(b'changed!');other.replace(path)
                        else:
                            path.write_bytes(b'changed!')
                            if mutation=='restore':path.write_bytes(b'original')
                    reached.append('private-key-generation')
                self.assertEqual(reached,[])

    def test_initial_corruption_or_missing_file_never_enters_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'public-model';path.write_bytes(b'changed!')
            for exists in [True,False]:
                if not exists:path.unlink()
                with self.assertRaises((ValueError,FileNotFoundError)):
                    with pinned_source(directory,{'public-model':hashlib.sha256(b'original').hexdigest()}):self.fail('unverified source materialized')


if __name__=='__main__':unittest.main()
