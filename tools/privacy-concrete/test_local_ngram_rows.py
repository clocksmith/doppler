"""Local storage boundary tests, using tiny public raw BF16 byte fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from local_ngram_rows import LocalNgramRows


class LocalRowsTests(unittest.TestCase):
    def fixture(self,directory):
        files={};parts=[];expected=bytearray();row=0
        for index,rows in enumerate([2,3,1]):
            name=f'public-{index}.safetensors';tensor=f'table.shard_{index}.weight'
            payload=bytes(range(row*4,(row+rows)*4));expected.extend(payload)
            header=json.dumps({tensor:dict(dtype='BF16',shape=[rows,2],data_offsets=[0,len(payload)])},sort_keys=True).encode()
            data=len(header).to_bytes(8,'little')+header+payload;(Path(directory)/name).write_bytes(data)
            files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),headerBytes=len(header),headerSha256=hashlib.sha256(header).hexdigest())
            parts.append(dict(name=tensor,shard=name,firstRow=row,rows=rows,columns=2,dtype='BF16',dataOffsets=[0,len(payload)]));row+=rows
        spec=dict(schema='doppler.flash-local-ngram-storage-inventory/v1',logicalShape=[row,2],logicalBytes=row*4,shards=files,parts=parts,wholeShardBytes=sum(p['bytes'] for p in files.values()))
        return spec,bytes(expected)

    def test_cross_shard_repeated_rows_match_one_local_concatenated_table(self):
        with tempfile.TemporaryDirectory() as directory:
            spec,all_rows=self.fixture(directory);indices=[5,0,2,1,4,2]
            with LocalNgramRows(directory,spec,local_storage_limit_bytes=4096) as storage:
                spec['parts'][0]['firstRow']=99 # caller cannot rewrite the bound layout
                self.assertEqual(storage.read_rows(indices,max_output_bytes=24),b''.join(all_rows[i*4:(i+1)*4] for i in indices))
                self.assertEqual(set(storage.__dict__) & {'indices','results','request_cache'},set())
            self.assertTrue(storage.closed);self.assertEqual(storage.maps,{})

    def test_cancelled_or_wrong_namespace_lease_cannot_close_other_requests_public_storage(self):
        from dataclasses import replace
        from flash_request_state import ClientContext
        with tempfile.TemporaryDirectory() as directory:
            spec,all_rows=self.fixture(directory)
            context=ClientContext('tenant','privacy','model','key','request')
            with LocalNgramRows(directory,spec,local_storage_limit_bytes=4096) as storage:
                first=storage.lease(context);other_context=replace(context,tenant_id='other',key_id='other',request_id='other')
                other=storage.lease(other_context);event=threading.Event();event.set()
                with self.assertRaises(AttributeError):first.context=other_context
                with self.assertRaises(AttributeError):first.storage=other
                with self.assertRaises(InterruptedError):first.read_rows([0],context,max_output_bytes=4,cancelled=event)
                self.assertTrue(first.closed);self.assertFalse(storage.closed)
                self.assertEqual(other.read_rows([5],other_context,max_output_bytes=4),all_rows[20:24])
                with self.assertRaises(ValueError):other.read_rows([0],context,max_output_bytes=4)
                self.assertTrue(other.closed);self.assertFalse(storage.closed)

    def test_external_inventory_snapshot_cannot_rewrite_the_bound_table_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            spec,all_rows=self.fixture(directory)
            with LocalNgramRows(directory,spec,local_storage_limit_bytes=4096) as storage:
                exposed=storage.spec;exposed['parts'][0]['dataOffsets']=[999,999]
                exposed['shards']['public-0.safetensors']['headerBytes']=0
                self.assertEqual(storage.spec,spec)
                self.assertEqual(storage.read_rows([0],max_output_bytes=4),all_rows[:4])

    def test_missing_corrupt_or_oversized_material_rejects_before_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            spec,_=self.fixture(directory)
            with self.assertRaises(ValueError):LocalNgramRows(directory,spec,local_storage_limit_bytes=1)
            wrong=copy.deepcopy(spec);wrong['parts'][1]['firstRow']=1
            with self.assertRaises(ValueError):LocalNgramRows(directory,wrong,local_storage_limit_bytes=4096)
            path=Path(directory)/'public-2.safetensors';path.write_bytes(path.read_bytes()[:-1]+b'x')
            with self.assertRaises(ValueError):LocalNgramRows(directory,spec,local_storage_limit_bytes=4096)
            path.unlink()
            with self.assertRaises(FileNotFoundError):LocalNgramRows(directory,spec,local_storage_limit_bytes=4096)

    def test_cancellation_mutation_and_invalid_indices_close_storage_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            for operation in ['cancel','mutate','bool-index','over-budget']:
                spec,_=self.fixture(directory);storage=LocalNgramRows(directory,spec,local_storage_limit_bytes=4096)
                event=threading.Event();indices=[0];budget=4
                if operation=='cancel':event.set()
                if operation=='mutate':
                    path=Path(directory)/'public-2.safetensors';path.write_bytes(path.read_bytes()[:-1]+b'x')
                if operation=='bool-index':indices=[True]
                if operation=='over-budget':budget=1
                with self.assertRaises((ValueError,InterruptedError)):storage.read_rows(indices,max_output_bytes=budget,cancelled=event)
                self.assertTrue(storage.closed);self.assertEqual(storage.maps,{})
                with self.assertRaises(ValueError):storage.read_rows([0],max_output_bytes=4)


if __name__=='__main__':unittest.main()
