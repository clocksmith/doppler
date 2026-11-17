"""Checked bridge against real twelve-bit NTT ciphertexts; transport is in-process.

Opt-in: needs the exact installed native NTT build. Exercises the accepted and
rejected response boundary with actual encryption, evaluation and decryption.
It does not qualify a model, a network deployment or malicious security.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

NATIVE=os.environ.get('DOPPLER_CONCRETE_NATIVE')=='1'
PROFILE='glwe-quant12-message35-storage48/v1'


@unittest.skipUnless(NATIVE,'opt-in native NTT ciphertext test')
class CheckedNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import concrete_ml_extensions as ext
        from native_runtime import ntt_source_id
        if ext.is_cuda_enabled() or ext.__doppler_ntt_build_id__!=ntt_source_id():raise ValueError('Exact installed CPU NTT source required')
        cls.ext=ext
        raw=json.loads(Path(__file__).with_name('precision12-policy.json').read_text())['parameters']
        cls.parameters=ext.MatmulCryptoParameters.deserialize(json.dumps(raw))
        cls.secret,cls.key=ext.cpu_create_private_key(cls.parameters);ext.cpu_prepare_ntt_key(cls.key)
        cls.other_secret,cls.other_key=ext.cpu_create_private_key(cls.parameters);ext.cpu_prepare_ntt_key(cls.other_key)
        generator=np.random.default_rng(21)
        cls.weights=generator.integers(-2**11,2**11,size=(896,640),dtype=np.int64)
        cls.inputs=generator.integers(0,2**12,size=(1,896),dtype=np.int64)
        cls.other_inputs=generator.integers(0,2**12,size=(1,896),dtype=np.int64)
        cls.exact=cls.inputs@cls.weights
        cls.cipher=cls.encrypt(cls.inputs)
        cls.good=cls.evaluate(cls.cipher,cls.weights)

    @classmethod
    def encrypt(cls,values,secret=None):
        return cls.ext.encrypt_matrix(pkey=cls.secret if secret is None else secret,crypto_params=cls.parameters,data=values.astype(np.uint64))

    @classmethod
    def evaluate(cls,cipher,weights,key=None):
        return cls.ext.cpu_matrix_multiplication_ntt(cipher,weights.astype(np.uint64),cls.key if key is None else key).serialize()

    def bridge(self,response,sampled=False):
        import checked_bridge
        from full_admission import FullRouteAdmission
        from full_plan import FullPlan,canonical
        from protocol import digest
        class CheckedPlan(FullPlan):
            @property
            def protocol_id(self):return 'test-linear-check/v1'
        row=dict(name='public-test',weightId=digest(self.weights.astype('<i2').tobytes()),inputWidth=896,outputWidth=640)
        plan=CheckedPlan(canonical(dict(modelId='a'*64,steps=2,contextBucket=1,inputTileWidth=2048,outputTileWidth=2048,
            parameters=dict(parameters=dict(bits_reserved_for_computation=35),quantizationBits=12,weightEncoding='signed-int12-in-int16'),projections=[row])))
        bridge=checked_bridge.CheckedBridge.__new__(checked_bridge.CheckedBridge)
        bridge.plan=plan;bridge.key=self.key;bridge.cancelled=None;bridge.sequence=1;bridge.weight_dtype='<i2'
        bridge.admission=FullRouteAdmission(plan);bridge.admission.admit_projection('public-test',896,640,1)
        tile=plan.schedule()[0];bridge.public_tile_digests={tile['id']:row['weightId']}
        bridge.remote_seconds=bridge.replay_seconds=0.;bridge.observe=lambda *args:None;bridge.child=SimpleNamespace(poll=lambda:0)
        events=[]
        def exchange(*args,while_pending=None):
            events.append('exchange')
            if while_pending is not None:while_pending()
            bridge.sequence+=1
            return response
        def replay(*args):
            events.append('replay');return self.ext.cpu_matrix_multiplication_ntt(*args)
        bridge.exchange=exchange;bridge.matrix_multiplication=replay
        bridge.begin_checks(PROFILE,bytes(range(32)))
        if not sampled:
            # Model a later tile of an already calibrated geometry outside the replay sample.
            bridge.result_lengths[(1,640)]=len(self.good);bridge.check.sampled=lambda tile_id,step:False
        return bridge,tile,events

    def attempt(self,response,sampled=False):
        bridge,tile,events=self.bridge(response,sampled)
        answer=bridge.evaluate(self.cipher,self.weights.astype(np.uint64),tile)
        decoded=bridge.decode(answer,tile,self.inputs.astype(np.uint64),
            lambda value:self.ext.decrypt_matrix(value,self.secret,self.parameters,640).astype(np.int64))
        bridge.check.retire()
        return bridge,decoded,events

    def assert_rejected(self,bridge,decoded,kind):
        from full_status import ExecutionRejected
        self.assertEqual([row['kind'] for row in bridge.failures],[kind]);self.assertFalse(decoded.any())
        with self.assertRaises(ExecutionRejected) as failure:bridge.require_integrity()
        self.assertEqual(failure.exception.code,'integrity-check-failed')

    def test_honest_native_response_is_accepted_without_replay(self):
        bridge,decoded,events=self.attempt(self.good)
        self.assertEqual(events,['exchange']);self.assertEqual(bridge.failures,[]);bridge.require_integrity()
        self.assertLess(int(np.abs(decoded-self.exact).max()),2000)

    def test_valid_ciphertexts_of_wrong_computations_are_rejected(self):
        zero=self.evaluate(self.cipher,np.zeros_like(self.weights))
        column=self.weights.copy();column[:,17]=0
        other_input=self.evaluate(self.encrypt(self.other_inputs),self.weights)
        other_key=self.evaluate(self.encrypt(self.inputs,self.other_secret),self.weights,self.other_key)
        for response in [zero,self.evaluate(self.cipher,column),other_input,other_key]:
            self.assertEqual(len(response),len(self.good))
            bridge,decoded,events=self.attempt(response)
            self.assertEqual(events,['exchange']);self.assert_rejected(bridge,decoded,'linear-check')

    def test_wrong_length_is_rejected_before_the_native_parser(self):
        from unittest.mock import patch
        import checked_bridge
        for response in [self.good[:-1],self.good+b'\0',b'']:
            with patch.object(checked_bridge.ext,'CompressedResultEncryptedMatrix',SimpleNamespace(deserialize=lambda raw:self.fail('wrong-length response parsed'))):
                bridge,decoded,_=self.attempt(response)
            self.assert_rejected(bridge,decoded,'response-length')

    def test_every_accepted_bit_flip_stays_inside_the_claimed_deviation(self):
        from linear_check import detection_threshold,noise_model
        threshold=detection_threshold(640,noise_model(PROFILE),log2_bound=40)
        generator=np.random.default_rng(22);accepted=rejected=0
        positions=sorted({int(value) for value in generator.integers(0,len(self.good)*8,size=400)}|set(range(0,64)))
        for position in positions:
            changed=bytearray(self.good);changed[position//8]^=1<<(position%8)
            bridge,decoded,_=self.attempt(bytes(changed))
            if bridge.failures:rejected+=1
            else:
                accepted+=1
                self.assertLess(int(np.abs(decoded-self.exact).max()),threshold)
        self.assertGreater(rejected,0)
        print(json.dumps(dict(event='native-bit-flips',positions=len(positions),accepted=accepted,rejected=rejected,threshold=threshold)))

    def test_noise_sized_wrong_coefficient_passes_the_linear_check_and_fails_sampled_replay(self):
        # Documented gap: one coefficient off by one shifts one output by at most 4095.
        nudged=self.weights.copy();nudged[5,9]+=1 if nudged[5,9]<2**11-1 else -1
        response=self.evaluate(self.cipher,nudged)
        bridge,decoded,_=self.attempt(response)
        self.assertEqual(bridge.failures,[]);self.assertLessEqual(int(np.abs(decoded-self.exact).max()),4095+2000)
        bridge,decoded,events=self.attempt(response,sampled=True)
        self.assertEqual(events,['exchange','replay']);self.assert_rejected(bridge,decoded,'sampled-replay-mismatch')

    def test_sampled_replay_accepts_the_exact_native_response(self):
        bridge,decoded,events=self.attempt(self.good,sampled=True)
        self.assertEqual(events,['exchange','replay']);self.assertEqual(bridge.failures,[]);self.assertEqual(bridge.sampled_replays,1)
        self.assertEqual(bridge.result_lengths[(1,640)],len(self.good))


if __name__=='__main__':unittest.main()
