"""Prefix control/ownership contracts; mocked dispatch does not prove encryption."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from flash_attention_prefix_probe import POLICY,POLICY_ID,require_prefix
from flash_text_rebuild5_plan import CONFIG


class PlanFixture:
    protocol_id=CONFIG['protocol']
    def __init__(self):
        self.rows=dict(schema=CONFIG['planSchema'],contextBucket=5,
            projections=[dict(name=name) for name in POLICY['projections']]+[dict(name='later')])
        counts=(5,5,5,10,6,2,2,6)
        self.tiles=tuple(dict(name=name,id=str(index)) for name,count in zip(POLICY['projections'],counts) for index in range(count))+(dict(name='later',id='next'),)
    def definition(self):return copy.deepcopy(self.rows)
    def schedule(self):return self.tiles


class PrefixContractTests(unittest.TestCase):
    def test_installed_policy_is_immutable_including_vectors(self):
        with self.assertRaises(TypeError):POLICY['maxAbsoluteError']=1
        with self.assertRaises(TypeError):POLICY['promptTokens'][0]=1
        with self.assertRaises(ValueError):require_prefix(PlanFixture(),POLICY['promptTokens'],expected_policy_id='0'*64)

    def test_complete_installed_prefix_and_public_fixture_only(self):
        self.assertEqual(require_prefix(PlanFixture(),POLICY['promptTokens']),41)
        for change in ('tokens','schema','protocol','rows','projection','tile-order','missing','late-duplicate'):
            with self.subTest(change=change):
                plan=PlanFixture();tokens=list(POLICY['promptTokens'])
                if change=='tokens':tokens[0]=True
                if change=='schema':plan.rows['schema']='legacy'
                if change=='protocol':plan.protocol_id='other'
                if change=='rows':plan.rows['contextBucket']=1
                if change=='projection':plan.rows['projections'][0]['name']='other'
                if change=='tile-order':plan.tiles=(plan.tiles[-1],*plan.tiles[:-1])
                if change=='missing':plan.tiles=plan.tiles[:-2]+plan.tiles[-1:]
                if change=='late-duplicate':plan.tiles+=plan.tiles[:1]
                with self.assertRaises(ValueError):require_prefix(plan,tokens)

    def test_cli_requires_public_fixture_before_run_or_prompt_read(self):
        from flash_generate_run import main
        args=['--source','/unread/source','--reference','/unread/reference','--scratch','/unread/scratch',
            '--backend-profile',POLICY['backendProfile'],'--tenant-id','tenant','--privacy-id','restricted',
            '--lock','/unread/lock','--pause-file','/unread/pause','--resource-wait-seconds','1','--attention-prefix-only']
        with patch('flash_generate_run.run') as run,contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(args),1);run.assert_not_called()

    def test_partial_prefix_never_marks_full_progress_validated(self):
        from flash_generate_run import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scratch=root/'scratch';scratch.mkdir();source=root/'source';source.mkdir()
            def run(*args,**kwargs):
                self.assertTrue(kwargs['attention_prefix_only']);progress=kwargs['progress'];progress.bind('a'*64,603472)
                for sequence in range(1,42):progress.observe('received-tile',dict(sequence=sequence))
                return dict(fullRouteCompleted=False,fullFlashModelExecuted=False)
            args=['--source',str(source),'--reference','/unread/reference','--scratch',str(scratch),
                '--backend-profile',POLICY['backendProfile'],'--tenant-id','tenant','--privacy-id','restricted',
                '--lock','/unread/lock','--pause-file','/unread/pause','--resource-wait-seconds','1',
                '--attention-prefix-only','--public-fixture',POLICY['publicFixtureId'],'--status-file',str(scratch/'status')]
            with patch('flash_generate_run.run',side_effect=run),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(args),0)
            rows=[json.loads(line) for line in (scratch/'status').read_text().splitlines()]
            self.assertEqual([row['phase'] for row in rows],['planned'])


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in installed Torch/source interfaces')
class PrefixLifecycleTests(unittest.TestCase):
    def exercise(self,failure=None):
        import torch
        from torch import nn
        from flash_attention_prefix_probe import execute_attention_prefix
        from flash_request_state import ClientContext
        from full_status import ExecutionRejected
        events=[];states=[]
        class Graph(nn.Module):
            def __init__(self):
                super().__init__();self.order=tuple(POLICY['projections']);self.identity='model'
                layer=nn.Module();layer.mlp_hyper_connection=nn.Identity();self.layers=nn.ModuleList([layer])
            def new_state(self,context):
                state=SimpleNamespace(position=0,context=context,closed=False)
                def close():state.closed=True;events.append('state-close')
                state.close=close;states.append(state);return state
            def forward(self,value,state,context,**kwargs):
                state.position=len(POLICY['projections'])
                shape=[1,5,8] if failure=='shape' else POLICY['outputShape']
                offset=2 if failure=='numerical' and context.key_id=='client-prefix-quantized' else 0
                return self.layers[0].mlp_hyper_connection(torch.ones(shape,dtype=torch.float32)+offset)
        graph=Graph()
        class Bridge:
            def __init__(self):
                self.plan=PlanFixture();self.cancelled=threading.Event();self.sequence=1;self.closed=False
                self.admission=SimpleNamespace(position=0,in_flight=False,projection=None,failed=False)
                self.child=SimpleNamespace(poll=lambda:0 if self.closed else None)
                self.sent=10;self.received=20;self.public_coefficient_bytes_sent=0;self.remote_seconds=.1;self.replay_seconds=.01
            def check_active(self):
                if self.cancelled.is_set():raise InterruptedError()
            def close(self):
                events.append('provider-close');self.closed=True;self.admission.failed=True
                if failure=='revocation':self.cancelled.set()
            def require_integrity(self):
                assert self.closed;events.append('integrity')
                if failure=='integrity':raise ExecutionRejected('integrity-check-failed')
            def integrity_report(self):return dict(securityQualified=False)
        bridge=Bridge()
        class Prepared:
            def __init__(self):self.graph=graph
            def execute(self,value,state,context,bridge,secret,parameters):
                events.append('execute')
                if failure=='native':raise RuntimeError('fixture native failure')
                if failure=='policy-reload':
                    import flash_attention_prefix_probe as probe
                    from types import MappingProxyType
                    probe.POLICY=MappingProxyType(dict(POLICY,maxAbsoluteError=0))
                    probe.POLICY_ID='0'*64
                bridge.sequence=42;bridge.admission.position=40 if failure=='cursor' else 41
                return graph(value,state,context)
            def reference(self,value,state,context,**kwargs):
                assert bridge.closed;events.append('quantized');return graph(value,state,context)
        class Floating:
            def __init__(self):self.graph=graph;self.cancelled=bridge.cancelled
        prepared=Prepared();oracle=Floating();context=ClientContext('tenant','restricted','model','key','request')
        import flash_attention_prefix_probe as probe
        original_id=probe.POLICY_ID
        with patch('checked_bridge.CheckedBridge',Bridge),patch('flash_prepared_graph.PreparedFlashGraph',Prepared),patch('flash_float_reference.BoundedFloatingReference',Floating),patch.object(probe,'POLICY',probe.POLICY),patch.object(probe,'POLICY_ID',probe.POLICY_ID):
            call=lambda:execute_attention_prefix(prepared,POLICY['promptTokens'],bridge,object(),object(),context,
                floating_reference=oracle,expected_policy_id=original_id)
            if failure and failure!='policy-reload':
                with self.assertRaises((ValueError,RuntimeError,InterruptedError)):call()
            else:
                result=call();self.assertEqual(result['encryptedTileCount'],41)
                self.assertEqual(result['maxEncryptedVsQuantizedError'],0)
                self.assertTrue(result['providerClosedBeforeValidation']);self.assertFalse(result['fullRouteCompleted'])
                self.assertFalse(result['fullFlashModelExecuted']);self.assertEqual(result['generatedTokens'],0)
                self.assertFalse(result['productionBackendAvailable']);self.assertFalse(result['securityQualified'])
                self.assertEqual(result['probePolicyId'],original_id)
            self.assertTrue(bridge.closed);self.assertTrue(bridge.admission.failed)
            self.assertTrue(all(state.closed for state in states));self.assertFalse(graph.layers[0].mlp_hyper_connection._forward_pre_hooks)
            if 'quantized' in events:self.assertLess(events.index('provider-close'),events.index('quantized'))
            if failure in ('native','cursor','integrity','revocation','shape'):self.assertNotIn('quantized',events)

    def test_close_then_compare_complete_prefix_with_isolated_reference_states(self):self.exercise()
    def test_reload_during_execution_cannot_change_attempt_policy_or_receipt_identity(self):self.exercise('policy-reload')
    def test_failure_never_releases_validated_output_or_leaks_hooks_and_state(self):
        for failure in ('native','cursor','integrity','revocation','shape','numerical'):
            with self.subTest(failure=failure):self.exercise(failure)


if __name__=='__main__':unittest.main()
