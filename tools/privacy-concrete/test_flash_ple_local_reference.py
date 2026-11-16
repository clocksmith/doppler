"""Mapped-table PLE integration against the pinned source on complete tiny public files."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import test_flash_ple_reference as fixtures


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in pinned source/Torch reference')
class LocalPLETests(unittest.TestCase):
    def fixture(self,directory,lookup_budget=65536):
        from local_ngram_rows import LocalNgramRows
        from flash_ple import ClientPLE
        torch,original,reference,context,config,path=fixtures.PLEReferenceTests().fixture()
        table=original.model.ple_embedding.ngram_embedding.weight.detach().to(torch.bfloat16).to(torch.float32)
        with torch.no_grad():original.model.ple_embedding.ngram_embedding.weight.copy_(table)
        reference.load_state_dict({key.replace('.source.','.'):value for key,value in original.model.state_dict().items()})
        total,width=table.shape;boundaries=[0,total//3,total//2,total];files={};parts=[]
        for index,(begin,end) in enumerate(zip(boundaries,boundaries[1:])):
            name=f'public-table-{index}.safetensors';tensor=f'table.shard_{index}.weight'
            payload=table[begin:end].to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
            header=json.dumps({tensor:dict(dtype='BF16',shape=[end-begin,width],data_offsets=[0,len(payload)])},sort_keys=True).encode()
            data=len(header).to_bytes(8,'little')+header+payload;(Path(directory)/name).write_bytes(data)
            files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),headerBytes=len(header),headerSha256=hashlib.sha256(header).hexdigest())
            parts.append(dict(name=tensor,shard=name,firstRow=begin,rows=end-begin,columns=width,dtype='BF16',dataOffsets=[0,len(payload)]))
        spec=dict(schema='doppler.flash-local-ngram-storage-inventory/v1',logicalShape=[total,width],logicalBytes=total*width*2,
                  shards=files,parts=parts,wholeShardBytes=sum(v['bytes'] for v in files.values()))
        storage=LocalNgramRows(directory,spec,local_storage_limit_bytes=1024*1024)
        parameters={key.replace('.source.','.'):value for key,value in original.model.state_dict().items() if key!='ple_embedding.ngram_embedding.weight'}
        mapped=ClientPLE.from_local_rows(path,config,storage,parameters,context_limit=8,max_lookup_bytes=lookup_budget)
        return torch,mapped,reference,context,config,path,storage,parameters

    def test_prefill_and_four_eos_crossing_updates_match_official_source_with_shared_mapped_rows(self):
        from flash_ple import ORDER
        from flash_ple_state import PLERequestState
        with tempfile.TemporaryDirectory() as directory:
            torch,model,reference,context,config,path,storage,parameters=self.fixture(directory)
            try:
                values=torch.sin(torch.arange(42,dtype=torch.float32)*.11).reshape(1,7,6)
                tokens=torch.tensor([[1,2,3,0,4,5,6]],dtype=torch.int64);state=PLERequestState(context);trace=[]
                def project(operation,value):trace.append(operation['id']);return getattr(model.model,operation['id']).source(value)
                for end in [3,4,5,6,7]:
                    begin=0 if end==3 else end-1
                    actual=model(values[:,begin:end],tokens[:,begin:end],state,context,execution='verified-projections',verified_client_projection=project)
                    with torch.no_grad():expected=reference(values[:,:end],tokens[:,:end],cache=None)[:,begin:end]
                    torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
                self.assertEqual(trace,list(ORDER)*5);self.assertEqual(state.conv[(1,2)].tolist(),[[5,6]])
                self.assertEqual(model.model.ple_embedding.ngram_embedding.weight.numel(),0)
                self.assertIs(state.local_rows_lease.storage,storage)
                state.close();self.assertFalse(storage.closed)
            finally:storage.close()

    def test_one_request_cancellation_or_wrong_context_cannot_affect_another_shared_table_request(self):
        from flash_ple_state import PLERequestState
        with tempfile.TemporaryDirectory() as directory:
            torch,model,reference,context,config,path,storage,parameters=self.fixture(directory)
            try:
                value=torch.zeros((1,1,6));tokens=torch.tensor([[1]],dtype=torch.int64);event=threading.Event()
                failed=PLERequestState(context)
                def cancel(operation,value):
                    result=getattr(model.model,operation['id']).source(value);event.set();return result
                with self.assertRaises(InterruptedError):model(value,tokens,failed,context,execution='verified-projections',verified_client_projection=cancel,cancelled=event)
                self.assertTrue(failed.failed);self.assertEqual(failed.conv,{});self.assertIsNone(failed.local_rows_lease);self.assertFalse(storage.closed)
                other=replace(context,tenant_id='other',key_id='other',request_id='other');state=PLERequestState(other)
                result=model(value,tokens,state,other,execution='public-reference')
                self.assertTrue(torch.isfinite(result).all());self.assertFalse(storage.closed)
                with patch.object(storage,'_read_rows',side_effect=AssertionError('wrong namespace read private rows')):
                    with self.assertRaises(ValueError):model(value,tokens,PLERequestState(context),other,execution='public-reference')
            finally:storage.close()

    def test_parameter_ownership_and_bounded_lookup_reject_without_remote_fallback(self):
        from flash_ple import ClientPLE
        from flash_ple_state import PLERequestState
        with tempfile.TemporaryDirectory() as directory:
            torch,model,reference,context,config,path,storage,parameters=self.fixture(directory,lookup_budget=1)
            try:
                value=torch.zeros((1,1,6));tokens=torch.tensor([[1]],dtype=torch.int64);state=PLERequestState(context)
                with self.assertRaises(ValueError):model(value,tokens,state,context,execution='public-reference')
                self.assertTrue(state.failed);self.assertFalse(storage.closed)
                missing=dict(parameters);missing.pop('key_proj.weight')
                with self.assertRaises(ValueError):ClientPLE.from_local_rows(path,config,storage,missing,context_limit=8,max_lookup_bytes=65536)
                fresh=ClientPLE.from_local_rows(path,config,storage,parameters,context_limit=8,max_lookup_bytes=65536)
                saved=fresh.model.key_proj.source.weight.detach().clone();parameters['key_proj.weight'].zero_()
                torch.testing.assert_close(fresh.model.key_proj.source.weight,saved,rtol=0,atol=0)
            finally:storage.close()


if __name__=='__main__':unittest.main()
