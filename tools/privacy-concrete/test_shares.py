import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np=None

from full_status import ExecutionRejected
import shares


def seeded(seed):
    state={'counter':0}
    def rng(count):
        state['counter']+=1
        return hashlib.shake_256(seed+state['counter'].to_bytes(8,'little')).digest(count)
    return rng


class Local:
    """In-process evaluator with an optional hook that alters its answer."""
    def __init__(self,weights,alter=None):
        self.weights,self.alter,self.sent,self.received,self.views=weights,alter,0,0,[]
    def send(self,name,rows):
        self.views.append(rows.copy());self.pending=shares.ring_product(rows,self.weights[name])
    def receive(self,rows,width):
        answer=self.pending.copy()
        return answer if self.alter is None else self.alter(answer)
    def close(self):self.closed=True


SEED=bytes(range(32))


@unittest.skipIf(np is None,'NumPy is not installed in the dependency-free smoke job')
class SharesTest(unittest.TestCase):
    def setUp(self):
        state=np.seterr(over='ignore');self.addCleanup(np.seterr,**state)
        generator=np.random.default_rng(7)
        self.weights={'p':generator.integers(-2048,2048,size=(48,20),dtype=np.int16)}
        self.anchors={'p':shares.build_anchor(SEED,'p',self.weights['p'])}
        self.inputs=generator.integers(0,4096,size=(3,48),dtype=np.uint64)
        self.expected=self.inputs.astype(np.int64)@self.weights['p'].astype(np.int64)
        self.rounds=shares.POLICY['checkRounds']

    def route(self,alter_a=None,alter_b=None,weights_a=None,weights_b=None,**options):
        a,b=Local(weights_a or self.weights,alter_a),Local(weights_b or self.weights,alter_b)
        return shares.ShareRoute((a,b),SEED,self.anchors,**options),a,b

    def rejected(self,**options):
        route,_,_=self.route(**options)
        result=route.linear('p',self.inputs,20,12)
        return route.integrity_failed() and not result.any()

    def test_exact_product_including_extremes(self):
        route,_,_=self.route()
        np.testing.assert_array_equal(route.linear('p',self.inputs,20,12),self.expected)
        route.require_integrity()
        low={'p':np.full((48,20),-2048,dtype=np.int16)}
        route=shares.ShareRoute((Local(low),Local(low)),SEED,{'p':shares.build_anchor(SEED,'p',low['p'])})
        np.testing.assert_array_equal(route.linear('p',np.full((2,48),4095,dtype=np.uint64),20,12),np.full((2,20),48*4095*-2048,dtype=np.int64))
        np.testing.assert_array_equal(route.linear('p',np.zeros((2,48),dtype=np.uint64),20,12),np.zeros((2,20),dtype=np.int64))
        route.require_integrity()

    def test_neither_evaluator_view_depends_on_the_input(self):
        other=(self.inputs+np.uint64(17))%np.uint64(4096)
        probe=shares.probe(SEED,'p',48)
        a_first,b_first,_=shares.build_queries(self.inputs,probe,self.rounds,seeded(b'fixed'))
        a_second,_,_=shares.build_queries(other,probe,self.rounds,seeded(b'fixed'))
        np.testing.assert_array_equal(a_first,a_second)
        # B: for every input there is an equally likely first share that gives B exactly the same rows.
        base=seeded(b'fixed');calls={'count':0}
        def shifted(count):
            calls['count']+=1;data=base(count)
            if calls['count']>1:return data
            return (np.frombuffer(data,dtype='<u8').reshape(self.inputs.shape)+(other-self.inputs)).tobytes()
        _,b_second,_=shares.build_queries(other,probe,self.rounds,shifted)
        np.testing.assert_array_equal(b_first,b_second)
        # No row is the probe or the input in clear, and fresh randomness is drawn on every call.
        self.assertFalse(any(np.array_equal(row,probe) for row in np.concatenate([a_first,b_first])))
        self.assertFalse(np.array_equal(b_first[:3],self.inputs))
        route,a,b=self.route();route.linear('p',self.inputs,20,12);route.linear('p',self.inputs,20,12)
        self.assertFalse(np.array_equal(a.views[0],a.views[1]));self.assertFalse(np.array_equal(b.views[0],b.views[1]))

    def test_any_altered_answer_is_caught_and_replaced_by_zeros(self):
        rows=self.inputs.shape[0];total=(1+2*self.rounds)*rows
        generator=np.random.default_rng(11)
        for side in ('a','b'):
            for row in range(total):
                column,change=int(generator.integers(0,20)),np.uint64(int(generator.integers(1,1<<62)))
                def alter(answer,row=row,column=column,change=change):
                    answer[row,column]+=change;return answer
                route,_,_=self.route(**{'alter_'+side:alter})
                result=route.linear('p',self.inputs,20,12)
                self.assertFalse(result.any(),(side,row))
                self.assertTrue(route.integrity_failed())
                # The failure is recorded, not raised: the attempt continues to its fixed end.
                route.linear('p',self.inputs,20,12);route.finish()
                with self.assertRaises(ExecutionRejected) as caught:route.require_integrity()
                self.assertEqual(caught.exception.code,'integrity-check-failed')

    def test_one_dishonest_evaluator_cannot_guess_the_multipliers(self):
        rows=self.inputs.shape[0]
        def doubled(answer):
            answer[:rows]+=answer[:rows];return answer
        def assumes_multiplier_three(answer):
            answer[:rows,2]+=np.uint64(5);answer[rows:2*rows,2]+=np.uint64(15);answer[2*rows:3*rows,2]+=np.uint64(15);return answer
        def other_matrix(answer):
            return answer+np.uint64(1)
        for alter in (doubled,assumes_multiplier_three,other_matrix):
            self.assertTrue(self.rejected(alter_a=alter),alter.__name__)
            self.assertTrue(self.rejected(alter_b=alter),alter.__name__)
        wrong={'p':self.weights['p'].copy()};wrong['p'][3,4]+=1
        self.assertTrue(self.rejected(weights_a=wrong));self.assertTrue(self.rejected(weights_b=wrong))

    def test_the_range_check_alone_rejects_a_change_that_matches_every_multiplier(self):
        probe=shares.probe(SEED,'p',48)
        bound=shares.product_bound(48,12)
        for inputs in (self.inputs,np.zeros((3,48),dtype=np.uint64)):
            for change in (np.uint64(1<<63),np.uint64(1<<62),np.uint64(2*bound)):
                a,b,secrets=shares.build_queries(inputs,probe,self.rounds,seeded(b'range'))
                answer_a,answer_b=shares.ring_product(a,self.weights['p']),shares.ring_product(b,self.weights['p'])
                _,passed=shares.combine(answer_a,answer_b,secrets,self.anchors['p'],bound)
                self.assertTrue(passed)
                # The evaluator is handed the secret multipliers, so every consistency check passes.
                answer_a[0,0]+=change
                for k in range(self.rounds):answer_a[3*(1+k),0]+=secrets['alpha'][k][0,0]*change
                product,passed=shares.combine(answer_a,answer_b,secrets,self.anchors['p'],bound)
                self.assertFalse(passed,(int(change),bool(inputs.any())))

    def test_two_evaluators_making_the_same_deviation_are_caught(self):
        zeros=lambda answer:np.zeros_like(answer)
        self.assertTrue(self.rejected(alter_a=zeros,alter_b=zeros))
        generator=np.random.default_rng(3)
        wrong={'p':(self.weights['p']+generator.integers(-3,4,size=(48,20))).astype(np.int16)}
        self.assertTrue(self.rejected(weights_a=wrong,weights_b=wrong))
        single={'p':self.weights['p'].copy()};single['p'][3,4]+=1
        self.assertTrue(self.rejected(weights_a=single,weights_b=single))
        # Honest on the first block only, or on everything but the first block.
        rows=self.inputs.shape[0]
        def lazy_checks(answer):
            answer[rows:]=0;return answer
        def lazy_real(answer):
            answer[:rows]=0;return answer
        self.assertTrue(self.rejected(alter_a=lazy_checks,alter_b=lazy_checks))
        self.assertTrue(self.rejected(alter_a=lazy_real,alter_b=lazy_real))
        # A small structured change, as an evaluator without the secret probe would have to choose it.
        kernel=np.zeros((48,20),dtype=np.int64);kernel[0,0],kernel[1,0]=1,-1
        self.assertTrue(self.rejected(weights_a={'p':self.weights['p']+kernel},weights_b={'p':self.weights['p']+kernel}))

    def test_two_independently_dishonest_evaluators_are_caught(self):
        def bump(answer):
            answer[:,2]+=np.uint64(9);return answer
        def scale(answer):
            return answer*np.uint64(3)
        for alter_a,alter_b in ((bump,bump),(bump,scale),(scale,scale)):
            self.assertTrue(self.rejected(alter_a=alter_a,alter_b=alter_b))

    def test_admission_and_strength(self):
        route,_,_=self.route()
        with self.assertRaises(ValueError):route.linear('q',self.inputs,20,12)
        with self.assertRaises(ValueError):route.linear('p',self.inputs.astype(np.int64),20,12)
        with self.assertRaises(ValueError):route.linear('p',np.zeros((shares.POLICY['maximumRows']+1,48),dtype=np.uint64),20,12)
        with self.assertRaises(ValueError):shares.probe(b'short','p',48)
        self.assertEqual(shares.detection_log2(shares.product_bound(4864,12),2),2*(64-37))
        for change in (dict(checkRounds=0),dict(ringBits=32),dict(rejection='immediate'),dict(operationTimeoutSeconds=0)):
            with self.assertRaises(ValueError):shares.validate_policy({**shares.POLICY,**change})
        route.close()
        with self.assertRaises(ExecutionRejected):route.linear('p',self.inputs,20,12)

    def provider_directory(self,directory,file=None):
        data=self.weights['p'].astype('<i2').tobytes()
        identity=hashlib.sha256(data).hexdigest()
        (Path(directory)/(identity+'.i16')).write_bytes(data)
        manifest=dict(schema='doppler.two-server-shares-manifest/v1',projections=[dict(name='p',inputWidth=48,outputWidth=20,weightId=identity,file=file or identity+'.i16',dtype='<i2')])
        (Path(directory)/'shares-manifest.json').write_text(json.dumps(manifest))
        return identity,[sys.executable,str(Path(shares.__file__).with_name('shares_evaluator.py')),'--cache-dir',str(directory)]

    def process_route(self,command,**options):
        return shares.ShareRoute((shares.ProcessEvaluator(command,'a',**options),shares.ProcessEvaluator(command,'b',**options)),SEED,self.anchors)

    def test_two_real_evaluator_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            identity,command=self.provider_directory(directory)
            route=self.process_route(command)
            try:
                np.testing.assert_array_equal(route.linear('p',self.inputs,20,12),self.expected)
                np.testing.assert_array_equal(route.linear('p',self.inputs[:1],20,12),self.expected[:1])
                route.finish()
            finally:
                route.close()
            route.require_integrity()
            report=route.report()
            self.assertEqual(report['sharedProjectionCalls'],2)
            self.assertEqual(report['sentBytes'][0],report['sentBytes'][1])
            # A changed coefficient file is refused by the evaluator; the client reports a failed execution.
            (Path(directory)/(identity+'.i16')).write_bytes(bytes(48*20*2))
            route=self.process_route(command)
            try:
                with self.assertRaises(ExecutionRejected) as caught:route.linear('p',self.inputs,20,12)
                self.assertEqual(caught.exception.code,'execution-failed')
            finally:
                route.close()

    def test_evaluator_serves_only_files_inside_its_directory_and_a_stalled_one_is_cut_off(self):
        with tempfile.TemporaryDirectory() as outer:
            inner=Path(outer)/'cache';inner.mkdir()
            identity,_=self.provider_directory(outer)
            _,command=self.provider_directory(inner,file='../'+identity+'.i16')
            route=self.process_route(command)
            try:
                with self.assertRaises(ExecutionRejected):route.linear('p',self.inputs,20,12)
            finally:
                route.close()
        stalled=[sys.executable,'-c','import sys,time;sys.stdin.buffer.read(8);time.sleep(30)']
        route=self.process_route(stalled,timeout=1)
        try:
            with self.assertRaises(ExecutionRejected) as caught:route.linear('p',self.inputs,20,12)
            self.assertEqual(caught.exception.code,'execution-failed')
        finally:
            route.close()


if __name__=='__main__':
    unittest.main()
