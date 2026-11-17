"""Secret linear-check arithmetic and its stated acceptance bound; numpy only, no native crypto."""
import copy
import unittest
try:
    import numpy as np
    import linear_check
    from linear_check import CheckKey,POLICY,acceptance_bound,detection_threshold,honest_norm_bound,noise_model,tolerance
except ImportError: # The dependency-free smoke job has no NumPy; these tests need it.
    np=None

PROFILE='glwe-quant12-message35-storage48/v1'
TILE='a'*64


def key(seed=1,policy=None):return CheckKey(bytes([seed])*32,PROFILE,35,12,POLICY if policy is None else policy)


def fixture(generator,inputs=2048,outputs=2048,rows=1):
    x=generator.integers(0,2**12,size=(rows,inputs),dtype=np.int64)
    w=generator.integers(-2**11,2**11,size=(inputs,outputs),dtype=np.int64)
    return x,w,x@w


def honest_error(generator,shape,outputs):
    noise=noise_model(PROFILE)
    sigma=np.sqrt(noise['varianceBase']+noise['variancePerOutput']*outputs)
    return np.rint(generator.normal(0,sigma,size=shape)).astype(np.int64)


@unittest.skipIf(np is None,'NumPy is required for the linear-check arithmetic')
class LinearCheckTests(unittest.TestCase):
    def test_attempt_policy_and_reported_identity_are_immutable_snapshots(self):
        from check_policy import policy_identity
        original=copy.deepcopy(POLICY);check=key(policy=original)
        samples=[check.sampled(TILE,step) for step in range(64)]
        original['replaySample']['denominator']=64;original['noise'][PROFILE]['margin']=4
        self.assertEqual(check.policy['replaySample']['denominator'],16);self.assertEqual(check.noise['margin'],2)
        self.assertEqual(check.policy_id,linear_check.POLICY_ID);self.assertEqual(policy_identity(check.policy),check.policy_id)
        self.assertEqual([check.sampled(TILE,step) for step in range(64)],samples)
        for name,value in (('policy',original),('policy_id','f'*64),('noise',{}),('count',1),('weight',1),('computation_bits',40),('quantization_bits',8)):
            with self.subTest(name=name),self.assertRaises(AttributeError):setattr(check,name,value)
        with self.assertRaises(TypeError):check.policy['replaySample']['denominator']=64
        with self.assertRaises(TypeError):check.noise['margin']=4

    def test_vectors_are_deterministic_bounded_and_separated_by_secret_and_tile(self):
        first=key().vectors(TILE,640,0,2)
        self.assertEqual(first.shape,(2,640,POLICY['checkVectors']));self.assertTrue(np.array_equal(first,key().vectors(TILE,640,0,2)))
        self.assertGreaterEqual(int(first.min()),-2**15);self.assertLess(int(first.max()),2**15)
        self.assertFalse(np.array_equal(first,key(2).vectors(TILE,640,0,2)))
        self.assertFalse(np.array_equal(first,key().vectors('b'*64,640,0,2)))
        # Rows and steps never share vectors, so persistent tampering faces independent checks.
        self.assertFalse(np.array_equal(first[0],first[1]))
        self.assertFalse(np.array_equal(first,key().vectors(TILE,640,1,2)))
        self.assertFalse(np.array_equal(first[0],key().vectors(TILE,640,0,1)[0]))

    def test_expectation_is_the_exact_integer_product(self):
        generator=np.random.default_rng(1);_,w,_=fixture(generator,512,640)
        check=key();vectors=check.vectors(TILE,640,3,2);actual=check.expectation(TILE,w,3,2)
        self.assertEqual(actual.shape,(2,512,POLICY['checkVectors']))
        for row in range(2):self.assertTrue(np.array_equal(actual[row],w@vectors[row]))
        extreme=np.full((2048,2048),-2**11,dtype=np.int64)
        self.assertTrue(np.array_equal(check.expectation(TILE,extreme,0,1)[0],extreme@check.vectors(TILE,2048,0,1)[0]))

    def test_honest_noise_at_the_installed_model_is_accepted(self):
        generator=np.random.default_rng(2)
        for outputs in [2,128,640,2048]:
            x,w,exact=fixture(generator,2048,outputs,rows=4);check=key();expectation=check.expectation(TILE,w,0,4)
            for _ in range(200):
                self.assertTrue(check.passes(TILE,x,exact+honest_error(generator,exact.shape,outputs),expectation,0))

    def test_honest_noise_far_above_the_measured_scale_is_still_accepted(self):
        # The tolerance is built from a norm bound with margin, so honest noise
        # 1.25 times the fitted scale must not cause false rejection.
        generator=np.random.default_rng(3);x,w,exact=fixture(generator);check=key();expectation=check.expectation(TILE,w,0,1)
        for _ in range(200):
            self.assertTrue(check.passes(TILE,x,exact+(1.25*honest_error(generator,exact.shape,2048)).astype(np.int64),expectation,0))

    def test_deviation_at_the_stated_threshold_is_rejected(self):
        generator=np.random.default_rng(4);noise=noise_model(PROFILE)
        for outputs in [16,640,2048]:
            threshold=detection_threshold(outputs,noise,log2_bound=40)
            self.assertLessEqual(acceptance_bound(outputs,threshold,noise),2.0**-40)
            x,w,exact=fixture(generator,2048,outputs)
            for seed in range(1,60):
                check=key(seed);expectation=check.expectation(TILE,w,0,1)
                changed=exact+honest_error(generator,exact.shape,outputs);changed[0,int(generator.integers(outputs))]+=threshold
                self.assertFalse(check.passes(TILE,x,changed,expectation,0))

    def test_measured_single_check_acceptance_stays_under_the_claimed_bound(self):
        # Every vector is an independent single check; count how often one accepts.
        generator=np.random.default_rng(5);noise=noise_model(PROFILE);outputs=640
        x,w,exact=fixture(generator,2048,outputs);limit=tolerance(outputs,noise);weight=2**15
        for factor in [1,4,16]:
            deviation=factor*limit//2**10;accepted=trials=0
            for seed in range(120):
                check=CheckKey(seed.to_bytes(32,'big'),PROFILE,35,12);changed=exact.copy();changed[0,7]+=deviation
                values=check.deviations(TILE,x,changed,check.expectation(TILE,w,0,1),0)
                accepted+=int((np.abs(values)<=limit).sum());trials+=values.size
            bound=min(1.0,(2*limit/deviation+1)/(2*weight))
            self.assertLessEqual(accepted/trials,min(1.0,bound+4*np.sqrt(bound*(1-bound)/trials)+1e-9))

    def test_tampering_every_row_and_step_faces_independent_checks(self):
        # A fixed pattern that one row accepts about half the time must not be
        # accepted on eleven rows at anything like that rate.
        generator=np.random.default_rng(9);outputs=640;limit=tolerance(outputs,noise_model(PROFILE))
        x,w,exact=fixture(generator,2048,outputs,rows=8);pattern=np.zeros(outputs,dtype=np.int64);pattern[3]=int(1.02*limit/2**15)
        single=whole=0;trials=300
        for seed in range(trials):
            check=CheckKey(seed.to_bytes(32,'big'),PROFILE,35,12)
            first=check.passes(TILE,x,exact+pattern,check.expectation(TILE,w,0,8),0)
            later=all(check.passes(TILE,x[:1],exact[:1]+pattern,check.expectation(TILE,w,step,1),step) for step in range(1,4))
            single+=check.passes(TILE,x[:1],exact[:1]+pattern,check.expectation(TILE,w,5,1),5);whole+=first and later
        self.assertGreater(single,0);self.assertLess(whole,max(1,single//8))

    def test_wrong_matrix_wrong_input_and_garbage_are_rejected(self):
        generator=np.random.default_rng(6);x,w,exact=fixture(generator,896,896);check=key();expectation=check.expectation(TILE,w,0,1)
        other_input=generator.integers(0,2**12,size=x.shape,dtype=np.int64)
        other_matrix=w.copy();other_matrix[:,3]=0
        for wrong in [other_input@w,x@other_matrix,np.zeros_like(exact),generator.integers(-2**33,2**33,size=exact.shape,dtype=np.int64)]:
            self.assertFalse(check.passes(TILE,x,wrong,expectation,0))
        for value in [2**34,-2**34-1,np.iinfo(np.int64).min,np.iinfo(np.int64).max]:
            outside=exact.copy();outside[0,0]=value
            self.assertIsNone(check.deviations(TILE,x,outside,expectation,0));self.assertFalse(check.passes(TILE,x,outside,expectation,0))
        inside=exact.copy();inside[0,0]=-2**34
        self.assertIsNotNone(check.deviations(TILE,x,inside,expectation,0))

    def test_small_deviation_inside_the_tolerance_is_accepted_as_documented(self):
        # The check bounds deviations; it does not prove equality. A change the
        # size of honest noise is accepted, which is why sampled replay exists.
        generator=np.random.default_rng(7);x,w,exact=fixture(generator);check=key();expectation=check.expectation(TILE,w,0,1)
        changed=exact.copy();changed[0,5]+=50
        self.assertTrue(check.passes(TILE,x,changed,expectation,0))
        # Any deviation whose l1 norm is at most tolerance/2^15 passes with certainty.
        budget=tolerance(2048,noise_model(PROFILE))//2**15
        changed=exact.copy();changed[0,:16]+=budget//16
        for seed in range(1,40):
            other=key(seed);self.assertTrue(other.passes(TILE,x,changed,other.expectation(TILE,w,0,1),0))

    def test_sample_selection_is_secret_dependent_deterministic_and_near_its_rate(self):
        check=key();hits=sum(check.sampled(format(index,'064x'),index%4) for index in range(16000))
        self.assertTrue(700<hits<1300)
        self.assertEqual([check.sampled(TILE,step) for step in range(64)],[key().sampled(TILE,step) for step in range(64)])
        self.assertNotEqual([check.sampled(format(i,'064x'),0) for i in range(256)],[key(2).sampled(format(i,'064x'),0) for i in range(256)])

    def test_retired_secret_and_unsupported_domains_reject(self):
        check=key();check.retire()
        for call in [lambda:check.vectors(TILE,4,0,1),lambda:check.sampled(TILE,0)]:
            with self.assertRaises(ValueError):call()
        with self.assertRaises(ValueError):CheckKey(b'short',PROFILE,35,12)
        with self.assertRaises(ValueError):CheckKey(bytes(32),'unmeasured-profile',35,12)
        with self.assertRaises(ValueError):key().expectation(TILE,np.full((4,4),2**11,dtype=np.int64),0,1)
        with self.assertRaises(ValueError):key().expectation(TILE,np.zeros((4,4),dtype=np.float64),0,1)
        check=key();generator=np.random.default_rng(8);x,w,exact=fixture(generator,16,16)
        with self.assertRaises(ValueError):check.passes(TILE,x+2**12,exact,check.expectation(TILE,w,0,1),0)

    def test_policy_identity_and_validation(self):
        self.assertEqual(len(linear_check.POLICY_ID),64)
        for change in [dict(securityQualified=True),dict(productionBackendAvailable=True),dict(checkVectors=0),dict(checkVectors=1),dict(weightBits=8),
                       dict(rejection='immediate'),dict(replaySample=dict(numerator=2,denominator=1)),dict(replaySample=dict(numerator=0,denominator=16)),
                       dict(replaySample=dict(numerator=1,denominator=4096)),dict(vectorScope='tile')]:
            with self.assertRaises(ValueError):linear_check.validate_policy(dict(POLICY,**change))
        self.assertGreater(honest_norm_bound(2048,noise_model(PROFILE)),0)
        loose=copy.deepcopy(POLICY);loose['noise'][PROFILE]['margin']=1e9
        with self.assertRaises(ValueError):noise_model(PROFILE,loose)


if __name__=='__main__':unittest.main()
