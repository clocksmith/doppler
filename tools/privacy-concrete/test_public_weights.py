"""Cache mutation must not redefine the client replay circuit after admission."""
import copy
from contextlib import contextmanager
import io
from pathlib import Path
import struct
from types import SimpleNamespace
import tempfile
import unittest
from full_plan import tiles
from public_weights import bind_public_tiles,PreparedPublicTiles,PUBLIC_TILE_BINDING_INTERFACE
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

    def test_streamed_binding_uses_bounded_fragmented_reads_for_both_encodings(self):
        for width,parameters in [(1,{}),(2,dict(weightEncoding='signed-int12-in-int16',quantizationBits=12))]:
            values=list(range(-35,35));raw=bytes(range(70)) if width==1 else struct.pack('<70h',*values)
            row=dict(name='projection',weightId=digest(raw),inputWidth=7,outputWidth=10)
            definition=dict(projections=[row],inputTileWidth=3,outputTileWidth=4,parameters=parameters)
            reads=[]
            class Fragmented(io.BytesIO):
                def read(self,size=-1):
                    if not 0<size<=4*width:raise AssertionError('unbounded public coefficient read')
                    reads.append(size);return super().read(min(size,3))
            @contextmanager
            def source(public_row,encoding):
                self.assertEqual(set(public_row),set(row));self.assertEqual(encoding['bytes'],width)
                with self.assertRaises(TypeError):public_row['weightId']='changed'
                with Fragmented(raw) as handle:yield handle
            plan=SimpleNamespace(definition=lambda:definition);binding=PreparedPublicTiles(plan,source).digests_for(plan)
            for tile in tiles(row,3,4):
                expected=b''.join(raw[(index*10+tile['outputBegin'])*width:(index*10+tile['outputEnd'])*width] for index in range(tile['inputBegin'],tile['inputEnd']))
                self.assertEqual(binding[tile['id']],digest(expected))
            self.assertTrue(reads);self.assertLessEqual(max(reads),4*width)

    def test_prepared_binding_survives_per_projection_eviction_and_rejects_other_plans(self):
        with tempfile.TemporaryDirectory() as parent:
            data={'first':bytes(range(24)),'second':bytes(reversed(range(24)))}
            rows=[dict(name=name,weightId=digest(raw),inputWidth=4,outputWidth=6) for name,raw in data.items()]
            definition=dict(modelId='a'*64,projections=rows,inputTileWidth=2,outputTileWidth=3);stages=[]
            @contextmanager
            def source(row,encoding):
                with tempfile.TemporaryDirectory(dir=parent) as directory:
                    path=Path(directory)/'owned';stages.append(path);path.write_bytes(data[row['name']])
                    with path.open('rb') as handle:yield handle
            plan=SimpleNamespace(definition=lambda:definition);catalog=PreparedPublicTiles(plan,source)
            self.assertEqual(catalog.interface,PUBLIC_TILE_BINDING_INTERFACE);self.assertTrue(all(not path.exists() for path in stages))
            values=catalog.digests_for(plan);self.assertEqual(len(values),8)
            with self.assertRaises(TypeError):values['untrusted']='replacement'
            changed=copy.deepcopy(definition);changed['modelId']='b'*64
            with self.assertRaises(ValueError):catalog.digests_for(SimpleNamespace(definition=lambda:changed))
            from plan_backends import load_backend_plan
            with self.assertRaises(ValueError):load_backend_plan('/unused',PUBLIC_TILE_BINDING_INTERFACE)

    def test_entire_route_identity_is_checked_before_any_coefficient_source_is_opened(self):
        calls=[];row=dict(name='valid',weightId='a'*64,inputWidth=4,outputWidth=6)
        bad=dict(row,name='bad',weightId='../../private-client-data')
        plan=SimpleNamespace(definition=lambda:dict(projections=[row,bad],inputTileWidth=2,outputTileWidth=3))
        def forbidden(*args):calls.append(args);raise AssertionError('source opened before complete route validation')
        with self.assertRaises(ValueError):PreparedPublicTiles(plan,forbidden)
        self.assertEqual(calls,[])
        plan=SimpleNamespace(definition=lambda:dict(projections=[row,row],inputTileWidth=2,outputTileWidth=3))
        with self.assertRaises(ValueError):PreparedPublicTiles(plan,forbidden)
        self.assertEqual(calls,[])

    def test_mixed_read_stream_never_becomes_an_admitted_catalog(self):
        raw=bytes(range(24));row=dict(name='projection',weightId=digest(raw),inputWidth=4,outputWidth=6)
        plan=SimpleNamespace(definition=lambda:dict(projections=[row],inputTileWidth=2,outputTileWidth=3))
        class Changed(io.BytesIO):
            calls=0
            def read(self,size=-1):
                data=super().read(size);self.calls+=1
                return bytes([99])*len(data) if self.calls==2 else data
        handles=[]
        @contextmanager
        def source(*args):
            with Changed(raw) as handle:handles.append(handle);yield handle
        with self.assertRaises(ValueError):PreparedPublicTiles(plan,source)
        self.assertTrue(handles[0].closed)

    def test_replaced_truncated_or_oversized_cache_cannot_rebind_identity(self):
        for content in [bytes([0])*24,b'short',bytes(25)]:
            with tempfile.TemporaryDirectory() as directory:
                path,row,plan=self.fixture(directory)
                temporary=path.with_suffix('.replacement');temporary.write_bytes(content);temporary.replace(path)
                with self.assertRaises(ValueError):bind_public_tiles(directory,plan)


if __name__=='__main__':unittest.main()
