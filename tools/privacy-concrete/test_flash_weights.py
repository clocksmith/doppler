"""Whole-object admission tests use tiny public fixtures, never real shard downloads."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from flash_weights import fetch_blob,verify_blob
from full_plan import FullPlan,canonical
from flash_expert_plan import FlashExpertPlan,POLICY_EXPERT,SOURCE_ID,WEIGHTS,load_plan
from protocol import POLICY


class Response(io.BytesIO):
    status=200
    headers={}
    def geturl(self):return 'https://public.invalid/model'


class WeightAdmissionTests(unittest.TestCase):
    def test_complete_object_is_atomic_and_cached_only_after_digest_validation(self):
        data=b'complete-public-source';record=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.safetensors'
            with patch('flash_weights.urllib.request.urlopen',return_value=Response(data)) as request:
                fetch_blob('https://public.invalid/model',path,record);verify_blob(path,record)
                fetch_blob('https://public.invalid/model',path,record)
                self.assertEqual(request.call_count,1)
            self.assertEqual(list(Path(directory).iterdir()),[path])
            path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):fetch_blob('https://public.invalid/model',path,record)

    def test_truncated_oversized_corrupt_and_partial_range_objects_never_publish(self):
        expected=b'public-source';record=dict(bytes=len(expected),sha256=hashlib.sha256(expected).hexdigest())
        for data,status,headers in [(b'bad',200,{}),(expected+b'!',200,{}),(b'X'*len(expected),200,{}),
                                    (expected,206,{'Content-Range':'bytes 0-12/99'})]:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'model.safetensors';response=Response(data);response.status=status;response.headers=headers
                with patch('flash_weights.urllib.request.urlopen',return_value=response),self.assertRaises(ValueError):
                    fetch_blob('https://public.invalid/model',path,record)
                self.assertEqual(list(Path(directory).iterdir()),[])

    def test_component_protocol_cannot_masquerade_as_full_qwen(self):
        component=FlashExpertPlan(canonical(dict(modelId='a'*64)))
        full=FullPlan(canonical(dict(modelId='a'*64)))
        self.assertNotEqual(component.protocol_id,full.protocol_id)
        self.assertNotEqual(component.frame('s','k',1,'evaluate','tile'),full.frame('s','k',1,'evaluate','tile'))
        from plan_backends import load_backend_plan
        with self.assertRaises(ValueError):load_backend_plan('/unused','unadvertised-flash-full-model')

    def test_incompatible_source_or_policy_fails_before_any_weight_access(self):
        values=dict(schema=POLICY_EXPERT['planSchema'],protocol=POLICY_EXPERT['protocol'],sourceId=SOURCE_ID,source=WEIGHTS,
                    productionBackendAvailable=False,implementationVersions={},parameters={key:POLICY[key] for key in ['dependencies','parameters','evaluationBackend']},
                    projections=[dict(name='expert/0/gate_up'),dict(name='expert/0/down')],
                    **{key:POLICY_EXPERT[key] for key in ['expertIndex','steps','contextBucket','inputTileWidth','outputTileWidth','cpuThreads','operationTimeoutSeconds']})
        for change in [dict(sourceId='b'*64),dict(steps=True),dict(expertIndex=True),dict(productionBackendAvailable=True)]:
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory)/'plan.json').write_bytes(canonical(dict(values,**change)))
                with patch('flash_expert_plan.versions',return_value={}),self.assertRaises(ValueError):load_plan(directory)


if __name__=='__main__':unittest.main()
