"""Checked-route ordering, deferral and rejection with real integer arithmetic.

Ciphertexts and transport are simulated; the admission cursor, public-weight
binding, linear checks, sampled replay and deferred rejection are the real code.
Native ciphertext behaviour is exercised separately by test_checked_native.py.
"""
import importlib
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
try:import numpy as np
except ImportError:np=None # The dependency-free smoke job has no NumPy.
from full_admission import FullRouteAdmission
from full_plan import FullPlan,canonical
from full_status import ExecutionRejected
from protocol import digest
from check_policy import POLICY_ID

PROFILE='glwe-quant12-message35-storage48/v1'
SEED=bytes(range(32))


class CheckedPlan(FullPlan):
    @property
    def protocol_id(self):return 'test-linear-check/v1'


class Cipher:
    def __init__(self,data):self.data=np.array(data,dtype=np.int64)
    def serialize(self):return self.data.tobytes()


@unittest.skipIf(np is None,'NumPy is required for the checked-bridge arithmetic')
class CheckedBridgeTests(unittest.TestCase):
    def test_unbound_or_mismatched_policy_rejects_before_base_key_or_provider_work(self):
        for binding in (None,'f'*64):
            definition=self.plan.definition()
            if binding is None:definition['parameters'].pop('integrityPolicyId')
            else:definition['parameters']['integrityPolicyId']=binding
            plan=CheckedPlan(canonical(definition))
            with self.subTest(binding=binding),patch.object(self.module.FullBridge,'__init__') as base:
                with self.assertRaisesRegex(ValueError,'policy digest'):
                    self.module.CheckedBridge('unused',plan,object(),parameter_profile=PROFILE)
                base.assert_not_called()

    def test_policy_change_during_base_preparation_closes_before_check_secret_or_payload(self):
        from check_policy import POLICY
        import copy
        original=copy.deepcopy(POLICY)
        def prepare(bridge,directory,plan,key,**kwargs):
            bridge.plan=plan;POLICY['replaySample']['denominator']=64
        try:
            with patch.object(self.module.FullBridge,'__init__',prepare),patch.object(self.module.CheckedBridge,'close') as close,patch.object(self.module.os,'urandom') as secret:
                with self.assertRaisesRegex(ValueError,'new compiled policy'):
                    self.module.CheckedBridge('unused',self.plan,object(),parameter_profile=PROFILE)
                close.assert_called_once();secret.assert_not_called()
        finally:POLICY.clear();POLICY.update(original)

    def test_attempt_cannot_replace_check_policy_and_reports_original_digest_after_global_change(self):
        from check_policy import POLICY
        import copy
        bridge,_,_=self.bridge();original=copy.deepcopy(POLICY);identity=bridge.check.policy_id
        try:
            POLICY['replaySample']['denominator']=64
            report=bridge.integrity_report()
            self.assertEqual(report['integrityPolicyId'],identity);self.assertEqual(report['replaySample']['denominator'],16)
            self.assertEqual(bridge.check.policy['replaySample']['denominator'],16)
            with self.assertRaises(ValueError):bridge.begin_checks(PROFILE,SEED)
        finally:POLICY.clear();POLICY.update(original)

    def setUp(self):
        self.parsed=[]
        def deserialize(raw):
            self.parsed.append(raw)
            if raw[:8]==b'\xff'*8:raise RuntimeError('native parser rejected the response')
            return raw
        native=SimpleNamespace(CompressedResultEncryptedMatrix=SimpleNamespace(deserialize=deserialize))
        with patch.dict(sys.modules,{'concrete_ml_extensions':native}):
            for name in ['full_bridge','checked_bridge']:sys.modules.pop(name,None)
            self.module=importlib.import_module('checked_bridge')
        generator=np.random.default_rng(11)
        self.weights={'first':generator.integers(-2**11,2**11,size=(64,48),dtype=np.int64),'second':generator.integers(-2**11,2**11,size=(48,64),dtype=np.int64)}
        rows=[dict(name=name,weightId=digest(matrix.astype('<i2').tobytes()),inputWidth=matrix.shape[0],outputWidth=matrix.shape[1]) for name,matrix in self.weights.items()]
        self.plan=CheckedPlan(canonical(dict(modelId='a'*64,steps=3,contextBucket=2,inputTileWidth=32,outputTileWidth=32,
            parameters=dict(parameters=dict(bits_reserved_for_computation=35),quantizationBits=12,weightEncoding='signed-int12-in-int16',integrityPolicyId=POLICY_ID),projections=rows)))
        self.generator=generator

    def bridge(self,tamper=None,event=None):
        bridge=self.module.CheckedBridge.__new__(self.module.CheckedBridge)
        bridge.plan=self.plan;bridge.admission=FullRouteAdmission(self.plan);bridge.cancelled=event;bridge.key=object();bridge.sequence=1
        bridge.weight_dtype='<i2';bridge.remote_seconds=bridge.replay_seconds=0.0;bridge.observe=lambda *args:None
        bridge.child=SimpleNamespace(poll=lambda:None);bridge.closed=False
        bridge.public_tile_digests={tile['id']:digest(np.ascontiguousarray(self.weights[tile['name']][tile['inputBegin']:tile['inputEnd'],tile['outputBegin']:tile['outputEnd']],dtype='<i2').tobytes()) for tile in self.plan.schedule()}
        wire=[];replays=[];history={}
        honest=lambda encrypted,matrix:(encrypted.data@matrix.view(np.int64)+self.noise(encrypted.data.shape[0],matrix.shape[1])).tobytes()
        self.noise_cache={}
        def evaluate(encrypted,matrix,key):
            replays.append(bridge.sequence-1);return SimpleNamespace(serialize=lambda:honest(encrypted,matrix))
        def exchange(kind,payload,response,tile_id,while_pending=None):
            tile=bridge.admission.schedule[(bridge.sequence-1)%len(bridge.admission.schedule)]
            wire.append((kind,len(payload),tile_id))
            matrix=np.ascontiguousarray(self.weights[tile['name']][tile['inputBegin']:tile['inputEnd'],tile['outputBegin']:tile['outputEnd']]).astype(np.uint64)
            encrypted=Cipher(np.frombuffer(payload,dtype=np.int64).reshape(-1,matrix.shape[0]))
            answer=honest(encrypted,matrix);history.setdefault(tile_id,[]).append(answer)
            if while_pending is not None:self.fail('checked bridge must not replay while a response is pending')
            index=bridge.sequence;bridge.sequence+=1
            return answer if tamper is None else tamper(index,tile,answer,history)
        bridge.matrix_multiplication=evaluate;bridge.exchange=exchange
        bridge.begin_checks(PROFILE,SEED)
        return bridge,wire,replays

    def noise(self,rows,outputs):
        # Deterministic "evaluation": replay and the simulated evaluator agree byte for byte.
        key=(rows,outputs)
        if key not in self.noise_cache:self.noise_cache[key]=np.rint(np.random.default_rng(5).normal(0,8,size=(rows,outputs))).astype(np.int64)
        return self.noise_cache[key]

    def run_route(self,bridge,close=True):
        """Drive the real cursor exactly as FullLinear does; return decoded outputs."""
        outputs=[];generator=np.random.default_rng(3)
        for step in range(3):
            rows=2 if step==0 else 1
            for name,matrix in self.weights.items():
                values=generator.integers(0,2**12,size=(rows,matrix.shape[0]),dtype=np.int64)
                with bridge.projection(name,matrix.shape[0],matrix.shape[1],rows):
                    result=np.zeros((rows,matrix.shape[1]),dtype=np.int64);encrypted={}
                    for tile in [tile for tile in self.plan.schedule() if tile['name']==name]:
                        encrypted.setdefault(tile['inputBegin'],Cipher(values[:,tile['inputBegin']:tile['inputEnd']]))
                        block=np.ascontiguousarray(matrix[tile['inputBegin']:tile['inputEnd'],tile['outputBegin']:tile['outputEnd']]).astype(np.uint64)
                        answer=bridge.evaluate(encrypted[tile['inputBegin']],block,tile)
                        width=tile['outputEnd']-tile['outputBegin']
                        result[:,tile['outputBegin']:tile['outputEnd']]+=bridge.decode(answer,tile,values[:,tile['inputBegin']:tile['inputEnd']].astype(np.uint64),
                            lambda raw:np.frombuffer(raw,dtype=np.int64).reshape(rows,width))
                outputs.append((values,result))
        bridge.admission.complete()
        if close:self.close(bridge)
        return outputs

    def close(self,bridge):
        bridge.child=SimpleNamespace(poll=lambda:0);bridge.check.retire()

    def test_honest_route_is_accepted_and_only_first_geometries_and_samples_replay(self):
        bridge,wire,replays=self.bridge();outputs=self.run_route(bridge)
        bridge.require_integrity()
        count=len(self.plan.schedule())*3
        self.assertEqual(len(wire),count);self.assertEqual(bridge.linear_checks,count);self.assertEqual(bridge.substituted_tiles,0)
        self.assertLess(len(replays),count);self.assertGreaterEqual(len(replays),4)
        for values,result in outputs[:1]:
            self.assertLess(int(np.abs(result-values@self.weights['first']).max()),200)

    def test_large_tamper_is_substituted_deferred_and_leaves_the_transcript_unchanged(self):
        honest,honest_wire,replayed=self.bridge();self.run_route(honest)
        target=next(index for index in range(5,len(honest_wire)) if index not in replayed)
        def tamper(index,tile,answer,history):
            if index!=target:return answer
            values=np.frombuffer(answer,dtype=np.int64).copy();values[0]+=10**7;return values.tobytes()
        bridge,wire,_=self.bridge(tamper);outputs=self.run_route(bridge,close=False)
        self.assertEqual(wire,honest_wire)
        self.assertEqual([row['kind'] for row in bridge.failures],['linear-check']);self.assertEqual(bridge.substituted_tiles,1)
        with self.assertRaises(ValueError):bridge.require_integrity()
        self.assertEqual(len(bridge.failures),1)
        self.close(bridge)
        with self.assertRaises(ExecutionRejected) as failure:bridge.require_integrity()
        self.assertEqual(failure.exception.code,'integrity-check-failed');self.assertTrue(bridge.admission.failed)

    def test_rejected_response_contributes_exactly_zero(self):
        honest,honest_wire,replayed=self.bridge();expected=self.run_route(honest)
        schedule=self.plan.schedule()
        # An unsampled prefill tile of the first projection, rejected by the linear check alone.
        target=next(index for index in range(1,len(schedule)+1) if index not in replayed and schedule[index-1]['name']=='first')
        tile=schedule[target-1]
        def tamper(index,tile_,answer,history):
            if index!=target:return answer
            return (np.frombuffer(answer,dtype=np.int64)+10**8).tobytes()
        bridge,_,_=self.bridge(tamper);actual=self.run_route(bridge)
        self.assertEqual([row['kind'] for row in bridge.failures],['linear-check'])
        values=expected[0][0];matrix=self.weights['first'][tile['inputBegin']:tile['inputEnd'],tile['outputBegin']:tile['outputEnd']]
        contribution=values[:,tile['inputBegin']:tile['inputEnd']]@matrix+self.noise(2,matrix.shape[1])
        without=expected[0][1].copy();without[:,tile['outputBegin']:tile['outputEnd']]-=contribution
        self.assertTrue(np.array_equal(actual[0][1],without))
        self.assertFalse(np.array_equal(actual[0][1],expected[0][1]))

    def test_recorded_failure_outranks_an_evaluator_chosen_transport_or_runtime_failure(self):
        from verified_execution import complete_verified_route
        for ending in ['finish','execute','cancel']:
            with self.subTest(ending=ending):
                honest,honest_wire,replayed=self.bridge();self.run_route(honest)
                target=next(index for index in range(5,len(honest_wire)) if index not in replayed)
                def tamper(index,tile,answer,history,target=target):
                    return answer if index!=target else (np.frombuffer(answer,dtype=np.int64)+10**8).tobytes()
                bridge,_,_=self.bridge(tamper)
                def finish():raise ValueError('Full evaluator failed to finish')
                def execute(step):
                    self.run_route(bridge,close=False)
                    if ending=='execute':raise ConnectionError('evaluator dropped the connection')
                    if ending=='cancel':raise InterruptedError('cancelled')
                bridge.finish=finish;bridge.close=lambda:self.close(bridge)
                with self.assertRaises(ExecutionRejected) as failure:complete_verified_route(bridge,1,execute,lambda step,result:None)
                self.assertEqual(failure.exception.code,'integrity-check-failed')
        # Without a recorded failure the original error is preserved.
        bridge,_,_=self.bridge();bridge.close=lambda:self.close(bridge)
        def finish():raise ValueError('Full evaluator failed to finish')
        bridge.finish=finish
        with self.assertRaises(ValueError) as failure:complete_verified_route(bridge,1,lambda step:self.run_route(bridge,close=False),lambda step,result:None)
        self.assertNotIsInstance(failure.exception,ExecutionRejected)

    def test_wrong_length_and_unparseable_responses_never_reach_decryption(self):
        decrypted=[]
        def tamper(index,tile,answer,history):
            if index==10:return answer+b'\0'
            if index==11:return b'\xff'*len(answer)
            return answer
        bridge,_,_=self.bridge(tamper)
        original=bridge.decode
        def decode(answer,tile,inputs,decrypt):
            return original(answer,tile,inputs,lambda raw:(decrypted.append(raw),decrypt(raw))[1])
        bridge.decode=decode;self.run_route(bridge)
        kinds=[row['kind'] for row in bridge.failures]
        self.assertIn('response-length',kinds)
        self.assertTrue(any(kind in ('response-parse','sampled-replay-mismatch') for kind in kinds))
        self.assertFalse(any(len(raw)%8 for raw in self.parsed));self.assertFalse(any(raw[:8]==b'\xff'*8 for raw in decrypted))
        with self.assertRaises(ExecutionRejected):bridge.require_integrity()

    def test_response_from_an_earlier_step_of_the_same_tile_is_rejected(self):
        schedule=len(self.plan.schedule())
        honest,honest_wire,replayed=self.bridge();self.run_route(honest)
        target=next(index for index in range(2*schedule+1,3*schedule+1) if index not in replayed)
        def tamper(index,tile,answer,history):
            # A step-2 single-row response replaced by step 1's response for the same tile.
            if index==target:return history[tile['id']][1]
            return answer
        bridge,_,_=self.bridge(tamper);self.run_route(bridge)
        self.assertEqual([row['kind'] for row in bridge.failures],['linear-check'])
        with self.assertRaises(ExecutionRejected):bridge.require_integrity()

    def test_noise_sized_change_passes_the_linear_check_but_not_a_sampled_replay(self):
        def tamper(index,tile,answer,history):
            values=np.frombuffer(answer,dtype=np.int64).copy();values[0]+=3;return values.tobytes()
        bridge,_,replays=self.bridge(tamper);self.run_route(bridge)
        count=len(self.plan.schedule())*3
        self.assertEqual({row['kind'] for row in bridge.failures},{'sampled-replay-mismatch'})
        self.assertEqual(len(bridge.failures),len(replays));self.assertEqual(bridge.linear_checks,count-len(replays))
        with self.assertRaises(ExecutionRejected):bridge.require_integrity()

    def test_cancellation_is_immediate_and_terminal(self):
        event=threading.Event()
        def tamper(index,tile,answer,history):
            if index==5:event.set()
            return answer
        bridge,wire,_=self.bridge(tamper,event)
        with self.assertRaises(InterruptedError):self.run_route(bridge)
        self.assertEqual(len(wire),5);self.assertTrue(bridge.admission.failed);event.clear()
        tile=self.plan.schedule()[0]
        with self.assertRaises(ValueError):bridge.evaluate(Cipher(np.zeros((1,32))),np.zeros((32,32),dtype=np.uint64),tile)
        self.assertEqual(len(wire),5)

    def test_public_weight_change_rejects_before_any_disclosure(self):
        bridge,wire,_=self.bridge();tile=self.plan.schedule()[0]
        matrix=np.ascontiguousarray(self.weights['first'][:32,:32]).astype(np.uint64);matrix[0,0]+=np.uint64(1)
        with self.assertRaises(ValueError):
            with bridge.projection('first',64,48,2):bridge.evaluate(Cipher(np.zeros((2,32))),matrix,tile)
        self.assertEqual(wire,[]);self.assertTrue(bridge.admission.failed)

    def test_decode_requires_the_pending_tile_and_evaluate_requires_decoding(self):
        bridge,_,_=self.bridge();tiles=self.plan.schedule()
        matrix=np.ascontiguousarray(self.weights['first'][:32,:32]).astype(np.uint64);values=np.zeros((2,32),dtype=np.uint64)
        with self.assertRaises(ValueError):
            with bridge.projection('first',64,48,2):
                bridge.evaluate(Cipher(values),matrix,tiles[0])
                bridge.decode(None,tiles[1],values,lambda raw:raw)
        self.assertTrue(bridge.admission.failed)

    def test_replay_protocol_plan_cannot_be_executed_with_the_checked_bridge(self):
        bridge=self.module.CheckedBridge.__new__(self.module.CheckedBridge)
        bridge.plan=FullPlan(self.plan.encoded)
        with self.assertRaises(ValueError):bridge.begin_checks(PROFILE,SEED)
        from full_ntt12_check_plan import POLICY_NTT12_CHECK
        from full_ntt12_plan import POLICY_NTT12
        self.assertTrue(POLICY_NTT12_CHECK['protocol'].endswith(self.module.CHECKED_PROTOCOL_SUFFIX))
        self.assertFalse(POLICY_NTT12['protocol'].endswith(self.module.CHECKED_PROTOCOL_SUFFIX))

    def test_check_secret_never_appears_on_the_wire_and_is_retired_at_close(self):
        payloads=[]
        bridge,_,_=self.bridge();original=bridge.exchange
        bridge.exchange=lambda kind,payload,*args,**kwargs:(payloads.append(payload),original(kind,payload,*args,**kwargs))[1]
        self.run_route(bridge)
        self.assertFalse(any(SEED in payload for payload in payloads));self.assertTrue(bridge.check.retired)
        report=bridge.integrity_report()
        self.assertEqual(report['integrity'],'secret-linear-check-sampled-replay/v1');self.assertFalse(report['securityQualified'])
        self.assertNotIn(SEED.hex(),str(report))


if __name__=='__main__':unittest.main()
