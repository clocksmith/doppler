"""Request-owned mapped input embedding fixtures; no actual Flash model claim."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest


def fixture_rows(directory):
    import torch
    from local_token_rows import LocalTokenRows,POLICY
    name='public-input.safetensors';tensor=POLICY['tensorName'];shape=[17,4]
    data=torch.linspace(-.1,.12,68).reshape(shape).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    header=json.dumps({tensor:dict(dtype='BF16',shape=shape,data_offsets=[0,len(data)])},separators=(',',':')).encode();header+=b' '*((-len(header))%8)
    blob=len(header).to_bytes(8,'little')+header+data;(Path(directory)/name).write_bytes(blob)
    inventory=dict(schema=POLICY['storageSchema'],logicalShape=shape,logicalBytes=len(data),wholeShardBytes=len(blob),
        shards={name:dict(bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),headerBytes=len(header),headerSha256=hashlib.sha256(header).hexdigest())},
        parts=[dict(name=tensor,shard=name,firstRow=0,rows=shape[0],columns=shape[1],dtype='BF16',dataOffsets=[0,len(data)])])
    return LocalTokenRows(directory,inventory,local_storage_limit_bytes=4096)


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in pinned Torch reference')
class TokenEmbeddingLeaseTests(unittest.TestCase):
    def test_ple_cleanup_does_not_close_a_different_requests_lease(self):
        from dataclasses import replace
        from flash_ple_state import PLERequestState
        from flash_request_state import ClientContext
        with tempfile.TemporaryDirectory() as directory:
            storage=fixture_rows(directory)
            try:
                context=ClientContext('tenant','privacy','model','key','request');other=replace(context,tenant_id='other')
                borrowed=storage.lease(other);state=PLERequestState(context);state.local_rows_lease=borrowed
                state.close();self.assertIsNone(state.local_rows_lease);self.assertFalse(borrowed.closed);self.assertFalse(storage.closed)
                borrowed.close()
            finally:storage.close()

    def test_cancelled_or_rebound_request_cannot_use_another_embedding_lease(self):
        import torch
        from dataclasses import replace
        from local_token_embedding import LocalTokenEmbedding
        from flash_client_projections import client_execution
        from flash_model_state import FlashModelState
        from flash_request_state import ClientContext
        with tempfile.TemporaryDirectory() as directory:
            storage=fixture_rows(directory)
            try:
                embedding=LocalTokenEmbedding(storage,max_lookup_rows=8);context=ClientContext('tenant','privacy','model','key','request')
                first=FlashModelState(context,(),(),'model');other=replace(context,tenant_id='other',request_id='other')
                second=FlashModelState(other,(),(),'model')
                with client_execution(first,context,(),'public-reference',None,None):embedding(torch.tensor([[1,2]],dtype=torch.int64));first.complete(2)
                with client_execution(second,other,(),'public-reference',None,None):embedding(torch.tensor([[1,2]],dtype=torch.int64));second.complete(2)
                first.token_embedding_lease=second.token_embedding_lease
                with self.assertRaises(ValueError):
                    with client_execution(first,context,(),'public-reference',None,None):embedding(torch.tensor([[3]],dtype=torch.int64))
                self.assertTrue(first.failed);self.assertFalse(storage.closed);self.assertFalse(second.token_embedding_lease.closed)
                # A distinct request always owns its own lease, including after the failed/rebound one.
                third=FlashModelState(context,(),(),'model')
                with client_execution(third,context,(),'public-reference',None,None):embedding(torch.tensor([[3]],dtype=torch.int64));third.complete(1)
                event=threading.Event();event.set()
                with self.assertRaises(InterruptedError):
                    with client_execution(third,context,(),'public-reference',None,event):embedding(torch.tensor([[3]],dtype=torch.int64))
                self.assertTrue(third.failed);self.assertFalse(storage.closed)
                for state in [first,second,third]:state.close()
            finally:storage.close()


if __name__=='__main__':unittest.main()
