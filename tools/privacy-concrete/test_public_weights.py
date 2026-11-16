"""Cache mutation must not redefine the client replay circuit after admission."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from full_plan import tiles
from public_weights import bind_public_tiles
from protocol import digest


class PublicWeightBindingTests(unittest.TestCase):
    def fixture(self,directory):
        data=bytes(range(24));identity=digest(data)
        path=Path(directory)/(identity+'.i8');path.write_bytes(data)
        row=dict(name='projection',weightId=identity,inputWidth=4,outputWidth=6)
        definition=dict(projections=[row],inputTileWidth=2,outputTileWidth=3)
        return path,row,SimpleNamespace(definition=lambda:definition)

    def test_strided_tiles_are_bound_to_owned_verified_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path,row,plan=self.fixture(directory)
            binding=bind_public_tiles(directory,plan)
            schedule=tiles(row,2,3)
            self.assertEqual(binding[schedule[0]['id']],digest(bytes([0,1,2,6,7,8])))
            self.assertEqual(binding[schedule[-1]['id']],digest(bytes([15,16,17,21,22,23])))
            # Same inode and length: a read-only mmap would change underneath its owner.
            path.write_bytes(bytes([99])*24)
            self.assertEqual(binding[schedule[0]['id']],digest(bytes([0,1,2,6,7,8])))
            self.assertNotEqual(binding[schedule[0]['id']],digest(bytes([99])*6))
            with self.assertRaises(ValueError):bind_public_tiles(directory,plan)

    def test_replaced_truncated_or_oversized_cache_cannot_rebind_identity(self):
        for content in [bytes([0])*24,b'short',bytes(25)]:
            with tempfile.TemporaryDirectory() as directory:
                path,row,plan=self.fixture(directory)
                temporary=path.with_suffix('.replacement');temporary.write_bytes(content);temporary.replace(path)
                with self.assertRaises(ValueError):bind_public_tiles(directory,plan)


if __name__=='__main__':unittest.main()
