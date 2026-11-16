"""Cache mutation must not redefine the client replay circuit after admission."""
import copy
from contextlib import contextmanager
import io
from pathlib import Path
import struct
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from full_plan import tiles
from public_weights import bind_public_tiles,PreparedPublicTiles,PUBLIC_TILE_BINDING_INTERFACE,PublicTileReader,store_public_matrix
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


class PublicTileReaderTests(unittest.TestCase):
    def fixture(self,directory,width=1,count=3):
        raw=bytes(range(35)) if width==1 else struct.pack('<35h',*range(-17,18))
        rows=[]
        for index in range(count):
            payload=raw if index%2==0 else raw[::-1]
            row=dict(name=f'projection-{index}',weightId=digest(payload),inputWidth=5,outputWidth=7)
            (Path(directory)/(row['weightId']+('.i8' if width==1 else '.i16'))).write_bytes(payload)
            rows.append(row)
        parameters={} if width==1 else dict(weightEncoding='signed-int12-in-int16',quantizationBits=12)
        definition=dict(projections=rows,inputTileWidth=2,outputTileWidth=3,parameters=parameters)
        return SimpleNamespace(definition=lambda:definition),rows,raw

    def test_owned_strided_tiles_and_one_handle_across_repeated_steps(self):
        for width in [1,2]:
            with tempfile.TemporaryDirectory() as directory:
                plan,rows,raw=self.fixture(directory,width);handles=[];opened=Path.open
                def track(path,*args,**kwargs):
                    self.assertTrue(all(handle.closed for handle in handles))
                    handle=opened(path,*args,**kwargs);handles.append(handle);return handle
                with patch.object(Path,'open',track),PublicTileReader(directory,plan) as reader:
                    for _step in range(2):
                        for index,row in enumerate(rows):
                            payload=raw if index%2==0 else raw[::-1]
                            for tile in tiles(row,2,3):
                                expected=b''.join(payload[(r*7+tile['outputBegin'])*width:(r*7+tile['outputEnd'])*width] for r in range(tile['inputBegin'],tile['inputEnd']))
                                actual=reader.read(tile);self.assertEqual(actual,expected)
                                self.assertIsInstance(actual,bytes)
                    self.assertEqual(len(handles),6)
                self.assertTrue(all(handle.closed for handle in handles))
                with self.assertRaises(ValueError):reader.read(tiles(rows[0],2,3)[0])

    def test_unknown_or_changed_geometry_rejects_before_open_and_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,rows,_raw=self.fixture(directory);tile=tiles(rows[0],2,3)[0]
            for changed in [dict(tile,name='private-file'),dict(tile,inputBegin=True),dict(tile,inputBegin=1),dict(tile,outputEnd=4),dict(tile,id='replacement'),dict(tile,weightId='../private')]:
                reader=PublicTileReader(directory,plan)
                with patch.object(Path,'open',side_effect=AssertionError('tile rejected after I/O')):
                    with self.assertRaises(ValueError):reader.read(changed)
                self.assertTrue(reader.closed)
                with self.assertRaises(ValueError):reader.read(tile)

    def test_truncated_missing_or_resized_sources_close_terminally(self):
        for operation in ['truncate-active','resize-before-open','missing']:
            with tempfile.TemporaryDirectory() as directory:
                plan,rows,_raw=self.fixture(directory);reader=PublicTileReader(directory,plan)
                path=Path(directory)/(rows[0]['weightId']+'.i8');schedule=tiles(rows[0],2,3)
                active=None
                if operation=='truncate-active':
                    reader.read(schedule[0]);active=reader.handle;path.write_bytes(b'')
                elif operation=='resize-before-open':path.write_bytes(b'short')
                else:path.unlink()
                with self.assertRaises((ValueError,FileNotFoundError)):reader.read(schedule[-1])
                self.assertTrue(reader.closed);self.assertIsNone(reader.handle)
                if active is not None:self.assertTrue(active.closed)

    def test_complete_identity_preflight_precedes_any_file_open(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,rows,_raw=self.fixture(directory)
            for bad in [dict(rows[-1],weightId='../../private'),dict(rows[-1],inputWidth=0),dict(rows[-1],name=rows[0]['name'])]:
                definition=copy.deepcopy(plan.definition());definition['projections'][-1]=bad
                with patch.object(Path,'open',side_effect=AssertionError('source opened in constructor')):
                    with self.assertRaises(ValueError):PublicTileReader(directory,SimpleNamespace(definition=lambda:definition))

    def test_exception_cleanup_removes_no_public_coefficient_files(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,rows,_raw=self.fixture(directory);paths=set(Path(directory).iterdir());active=None
            with self.assertRaises(RuntimeError):
                with PublicTileReader(directory,plan) as reader:
                    reader.read(tiles(rows[0],2,3)[0]);active=reader.handle
                    raise RuntimeError('cancelled owner')
            self.assertTrue(active.closed);self.assertEqual(set(Path(directory).iterdir()),paths)


class PublicMatrixStoreTests(unittest.TestCase):
    def store(self,directory,payload=b'abcdef',**changes):
        arguments=dict(name='public',input_width=2,output_width=3,parameters={},payload=payload)
        arguments.update(changes)
        return store_public_matrix(directory,**arguments)

    def test_content_addressed_install_reuse_and_explicit_regeneration(self):
        for payload,parameters,suffix in [(b'abcdef',{},'.i8'),(struct.pack('<6h',-2048,-17,0,1,127,2047),dict(weightEncoding='signed-int12-in-int16',quantizationBits=12),'.i16')]:
            with tempfile.TemporaryDirectory() as directory:
                path,row=self.store(directory,payload,parameters=parameters)
                self.assertEqual(path.name,digest(payload)+suffix);self.assertEqual(path.read_bytes(),payload)
                self.assertEqual(row,dict(name='public',inputWidth=2,outputWidth=3,weightId=digest(payload)))
                with patch('public_weights.tempfile.NamedTemporaryFile',side_effect=AssertionError('valid cache rewritten')):
                    self.assertEqual(self.store(directory,payload,parameters=parameters),(path,row))
                path.unlink() # Public preparation can explicitly recreate an evicted entry.
                self.assertEqual(self.store(directory,payload,parameters=parameters),(path,row))
                self.assertEqual(path.read_bytes(),payload);self.assertEqual(set(Path(directory).iterdir()),{path})

    def test_invalid_encoding_geometry_or_ownership_rejects_before_filesystem(self):
        for changes in [dict(input_width=True),dict(output_width=0),dict(name=''),dict(payload=bytearray(b'abcdef')),dict(payload=b'short'),dict(parameters=dict(weightEncoding='unqualified'))]:
            with patch('public_weights.tempfile.NamedTemporaryFile',side_effect=AssertionError('invalid cache wrote a file')),patch.object(Path,'exists',side_effect=AssertionError('invalid cache accessed filesystem')):
                with self.assertRaises(ValueError):self.store('/unused',**changes)

    def test_existing_corruption_and_symlinks_are_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path,row=self.store(directory);path.write_bytes(b'broken')
            with self.assertRaises(ValueError):self.store(directory)
            self.assertEqual(path.read_bytes(),b'broken')
            path.unlink();other=Path(directory)/'unrelated';other.write_bytes(b'abcdef');path.symlink_to(other)
            with self.assertRaises(ValueError):self.store(directory)
            self.assertTrue(path.is_symlink());self.assertEqual(other.read_bytes(),b'abcdef')

    def test_write_or_install_failure_cleans_only_its_owned_temporary_file(self):
        for failure in ['write','install']:
            with tempfile.TemporaryDirectory() as directory:
                sibling=Path(directory)/'unrelated';sibling.write_bytes(b'preserve')
                from tempfile import NamedTemporaryFile
                class InterruptedFile:
                    def __init__(self,*args,**kwargs):self.handle=NamedTemporaryFile(*args,**kwargs);self.name=self.handle.name
                    def __enter__(self):return self
                    def __exit__(self,*args):return self.handle.__exit__(*args)
                    def write(self,chunk):raise InterruptedError('cancelled public preparation')
                name='public_weights.tempfile.NamedTemporaryFile' if failure=='write' else 'public_weights.os.link'
                replacement=InterruptedFile if failure=='write' else lambda *args:(_ for _ in ()).throw(OSError('link rejected'))
                with patch(name,side_effect=replacement):
                    with self.assertRaises(OSError):self.store(directory)
                self.assertEqual(set(Path(directory).iterdir()),{sibling});self.assertEqual(sibling.read_bytes(),b'preserve')

    def test_concurrent_identical_writers_share_one_complete_cache_entry(self):
        from concurrent.futures import ThreadPoolExecutor
        import os
        import threading
        with tempfile.TemporaryDirectory() as directory:
            barrier=threading.Barrier(2);link=os.link
            def race(source,target):barrier.wait(timeout=5);return link(source,target)
            with patch('public_weights.os.link',side_effect=race),ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(self.store,directory) for _index in range(2)]
                results=[future.result(timeout=10) for future in futures]
            self.assertEqual(results[0],results[1]);path,_row=results[0]
            self.assertEqual(path.read_bytes(),b'abcdef');self.assertEqual(set(Path(directory).iterdir()),{path})

    def test_content_verification_accepts_only_harmless_link_metadata_changes(self):
        import hashlib
        import os
        with tempfile.TemporaryDirectory() as directory:
            path,_row=self.store(directory);original=hashlib.file_digest
            def verify(handle,algorithm):
                alias=Path(directory)/'owned-link';os.link(path,alias);alias.unlink()
                return original(handle,algorithm)
            with patch('public_weights.hashlib.file_digest',side_effect=verify):
                self.assertEqual(self.store(directory)[0],path)
            self.assertEqual(path.read_bytes(),b'abcdef')

    def test_conflicting_concurrent_entry_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            def conflict(source,target):Path(target).write_bytes(b'broken');raise FileExistsError('raced cache')
            with patch('public_weights.os.link',side_effect=conflict):
                with self.assertRaises(ValueError):self.store(directory)
            path=Path(directory)/(digest(b'abcdef')+'.i8')
            self.assertEqual(path.read_bytes(),b'broken');self.assertEqual(set(Path(directory).iterdir()),{path})


if __name__=='__main__':unittest.main()
