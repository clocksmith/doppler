import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import threading
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
    """In-process evaluator with an optional hook that alters its answer; the hook also sees the rows it was sent."""
    def __init__(self,weights,alter=None):
        self.weights,self.alter,self.sent,self.received,self.views=weights,alter,0,0,[]
    def send(self,name,rows):
        self.views.append(rows.copy());self.pending=shares.field_product(rows,self.weights[name])
    def receive(self,rows,width):
        answer=self.pending.copy()
        return answer if self.alter is None else self.alter(answer,self.views[-1])
    def close(self):self.closed=True


SEED=bytes(range(32))
P=shares.PRIME


@unittest.skipIf(np is None,'NumPy is not installed in the dependency-free smoke job')
class SharesTest(unittest.TestCase):
    def setUp(self):
        generator=np.random.default_rng(7)
        self.weights={'p':generator.integers(-2048,2048,size=(48,20),dtype=np.int16)}
        self.anchors={'p':shares.build_anchor(SEED,'p',self.weights['p'])}
        self.inputs=generator.integers(0,4096,size=(3,48),dtype=np.uint64)
        self.expected=self.inputs.astype(np.int64)@self.weights['p'].astype(np.int64)
        self.rounds=shares.POLICY['checkRounds']
        self.bound=shares.product_bound(48,12)

    def route(self,alter_a=None,alter_b=None,weights_a=None,weights_b=None,**options):
        a,b=Local(weights_a or self.weights,alter_a),Local(weights_b or self.weights,alter_b)
        return shares.ShareRoute((a,b),SEED,options.pop('anchors',self.anchors),**options),a,b

    def rejected(self,inputs=None,**options):
        route,_,_=self.route(**options)
        result=route.linear('p',self.inputs if inputs is None else inputs,20,12)
        return route.integrity_failed() and not result.any()

    def test_field_arithmetic_matches_exact_integers(self):
        generator=np.random.default_rng(1)
        left=np.array([int(generator.integers(0,P)) for _ in range(500)]+[0,1,P-1,P-1],dtype=np.uint64)
        right=np.array([int(generator.integers(0,P)) for _ in range(500)]+[P-1,P-1,P-1,1],dtype=np.uint64)
        for got,operation in ((shares.multiply(left,right),lambda u,v:u*v),(shares.add(left,right),lambda u,v:u+v),(shares.subtract(left,right),lambda u,v:u-v)):
            self.assertEqual(got.tolist(),[operation(int(u),int(v))%P for u,v in zip(left,right)])
        weights=generator.integers(-2048,2048,size=(48,9),dtype=np.int16);weights[0,0],weights[1,0]=-32768,32767
        rows=shares._field(generator.bytes(8*4*48),(4,48)).copy();rows[0,:]=P-1
        want=[[sum(int(rows[r,i])*int(weights[i,c]) for i in range(48))%P for c in range(9)] for r in range(4)]
        for chunk in (1,100,10**6):self.assertEqual(shares.field_product(rows,weights,chunk_elements=chunk).tolist(),want)

    def test_exact_product_including_extremes(self):
        route,_,_=self.route()
        np.testing.assert_array_equal(route.linear('p',self.inputs,20,12),self.expected)
        route.require_integrity()
        low={'p':np.full((48,20),-2048,dtype=np.int16)}
        route=shares.ShareRoute((Local(low),Local(low)),SEED,{'p':shares.build_anchor(SEED,'p',low['p'])})
        np.testing.assert_array_equal(route.linear('p',np.full((2,48),4095,dtype=np.uint64),20,12),np.full((2,20),48*4095*-2048,dtype=np.int64))
        np.testing.assert_array_equal(route.linear('p',np.zeros((2,48),dtype=np.uint64),20,12),np.zeros((2,20),dtype=np.int64))
        route.require_integrity()

    def test_neither_evaluator_view_depends_on_the_input_the_probe_or_the_multipliers(self):
        other=(self.inputs+np.uint64(17))%np.uint64(4096)
        probe,other_probe=shares.probe(SEED,'p',48),shares.probe(bytes(32),'p',48)
        self.assertFalse(np.array_equal(probe,other_probe))
        a_first,b_first,secrets=shares.build_queries(self.inputs,probe,2,seeded(b'fixed'))
        a_second,_,_=shares.build_queries(other,probe,2,seeded(b'fixed'))
        np.testing.assert_array_equal(a_first,a_second)
        # B: for every input there is an equally likely first share that gives B exactly the same rows.
        def shifted(base,index,change):
            calls={'count':0}
            def rng(count):
                calls['count']+=1;data=base(count)
                if calls['count']!=index:return data
                values=shares._field(data,change.shape)
                return shares.add(values,change).astype('<u8').tobytes()
            return rng
        _,b_second,_=shares.build_queries(other,probe,2,shifted(seeded(b'fixed'),1,shares.subtract(other,self.inputs)))
        np.testing.assert_array_equal(b_first,b_second)
        # A: a different probe gives the same rows under an equally likely choice of the masks t (the seventh draw).
        rows=3;change=np.concatenate([shares.multiply(secrets['delta'][k],shares.subtract(probe,other_probe).reshape(1,-1)) for k in range(2)]).reshape(2,rows,48)
        a_third,_,_=shares.build_queries(self.inputs,other_probe,2,shifted(seeded(b'fixed'),6,change))
        np.testing.assert_array_equal(a_first,a_third)
        # Structure: A's check rows are masked by t, which only B holds in clear; u differs from t; rounds differ.
        part=lambda view,index:view[index*rows:(index+1)*rows]
        for k in range(2):
            unmasked=shares.add(shares.multiply(secrets['alpha'][k],part(a_first,0)),shares.multiply(secrets['delta'][k],probe.reshape(1,-1)))
            np.testing.assert_array_equal(shares.subtract(part(a_first,1+k),unmasked),part(b_first,3+k))
            self.assertFalse(np.array_equal(part(a_first,3+k),part(b_first,3+k)))
        self.assertFalse(np.array_equal(part(b_first,3),part(b_first,4)))
        self.assertFalse(np.array_equal(part(a_first,3),part(a_first,4)))
        for name in ('alpha','beta','delta','epsilon'):
            self.assertGreater(int(secrets[name].max()),1<<50);self.assertEqual(len(np.unique(secrets[name])),secrets[name].size)
        self.assertTrue(bool((np.concatenate([a_first,b_first])<np.uint64(P)).all()))
        route,a,b=self.route();route.linear('p',self.inputs,20,12);route.linear('p',self.inputs,20,12)
        self.assertFalse(np.array_equal(a.views[0],a.views[1]));self.assertFalse(np.array_equal(b.views[0],b.views[1]))

    def test_one_evaluator_altering_any_block_is_caught_and_replaced_by_zeros(self):
        rows=self.inputs.shape[0];total=(1+2*self.rounds)*rows
        generator=np.random.default_rng(11)
        for side in ('a','b'):
            for row in range(total):
                column,change=int(generator.integers(0,20)),np.uint64(int(generator.integers(1,P)))
                def alter(answer,view,row=row,column=column,change=change):
                    answer[row,column]=shares.add(answer[row,column],change);return answer
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
        def doubled(answer,view):
            answer[:rows]=shares.add(answer[:rows],answer[:rows]);return answer
        def assumes_multiplier_three(answer,view):
            answer[:rows,2]=shares.add(answer[:rows,2],np.uint64(5));answer[rows:2*rows,2]=shares.add(answer[rows:2*rows,2],np.uint64(15));return answer
        def other_matrix(answer,view):
            return shares.add(answer,view[:,:1])
        def out_of_field(answer,view):
            answer[0,0]=np.uint64(P);return answer
        for alter in (doubled,assumes_multiplier_three,other_matrix,out_of_field):
            self.assertTrue(self.rejected(alter_a=alter),alter.__name__)
            self.assertTrue(self.rejected(alter_b=alter),alter.__name__)
        wrong={'p':self.weights['p'].copy()};wrong['p'][3,4]+=1
        self.assertTrue(self.rejected(weights_a=wrong));self.assertTrue(self.rejected(weights_b=wrong))

    def test_a_change_that_matches_every_multiplier_passes_the_checks_and_only_then_meets_the_bound(self):
        probe=shares.probe(SEED,'p',48)
        for change,accepted in ((1,True),(2*self.bound,False),(P-2*self.bound,False)):
            a,b,secrets=shares.build_queries(self.inputs,probe,self.rounds,seeded(b'bound'))
            answer_a,answer_b=shares.field_product(a,self.weights['p']),shares.field_product(b,self.weights['p'])
            product,passed=shares.combine(answer_a,answer_b,secrets,self.anchors['p'],self.bound)
            self.assertTrue(passed);np.testing.assert_array_equal(product,self.expected)
            # The evaluator is handed the secret multipliers, which it never has, so every field check passes.
            answer_a[0,0]=shares.add(answer_a[0,0],np.uint64(change))
            for k in range(self.rounds):answer_a[3*(1+k),0]=shares.add(answer_a[3*(1+k),0],shares.multiply(secrets['alpha'][k][0,0],np.uint64(change)))
            product,passed=shares.combine(answer_a,answer_b,secrets,self.anchors['p'],self.bound)
            self.assertEqual(passed,accepted,change)
            if accepted:self.assertEqual(int(product[0,0]),int(self.expected[0,0])+1)

    def test_two_evaluators_making_the_same_deviation_are_caught_whatever_the_input(self):
        zeros=lambda answer,view:np.zeros_like(answer)
        rows=self.inputs.shape[0]
        def lazy_checks(answer,view):
            answer[rows:]=0;return answer
        def lazy_real(answer,view):
            answer[:rows]=0;return answer
        generator=np.random.default_rng(3)
        dense={'p':(self.weights['p']+generator.integers(-3,4,size=(48,20))).astype(np.int16)}
        single={'p':self.weights['p'].copy()};single['p'][3,4]+=1
        kernel={'p':self.weights['p'].copy()};kernel['p'][0,0]+=1;kernel['p'][1,0]-=1
        even=self.inputs&~np.uint64(1);odd=self.inputs|np.uint64(1)
        for inputs in (self.inputs,even,odd,np.zeros_like(self.inputs)):
            for alter in (zeros,lazy_checks,lazy_real):
                self.assertTrue(self.rejected(inputs=inputs,alter_a=alter,alter_b=alter),alter.__name__)
            for wrong in (dense,single,kernel):
                self.assertTrue(self.rejected(inputs=inputs,weights_a=wrong,weights_b=wrong))
            # The same other matrix with one very large entry: W + c*E[5,2], for c with every power-of-two structure.
            for scale in (1,1<<30,1<<60,(1<<60)+1,P-1):
                def large(answer,view,scale=scale):
                    answer[:,2]=shares.add(answer[:,2],shares.multiply(view[:,5],np.uint64(scale)));return answer
                self.assertTrue(self.rejected(inputs=inputs,alter_a=large,alter_b=large),scale)

    def test_two_independently_dishonest_evaluators_are_caught(self):
        def bump(answer,view):
            answer[:,2]=shares.add(answer[:,2],np.uint64(9));return answer
        def scale(answer,view):
            return shares.multiply(answer,np.uint64(3))
        for alter_a,alter_b in ((bump,scale),(scale,scale),(scale,bump)):
            self.assertTrue(self.rejected(alter_a=alter_a,alter_b=alter_b))

    def test_the_probe_and_anchors_depend_on_the_secret_seed(self):
        stale={'p':shares.build_anchor(bytes(32),'p',self.weights['p'])}
        self.assertFalse(np.array_equal(stale['p'],self.anchors['p']))
        self.assertTrue(self.rejected(anchors=stale))
        for seed in (b'short',bytes(33)):
            with self.assertRaises(ValueError):shares.probe(seed,'p',48)
        # The width and the name are bound separately, so neither a prefix nor a shifted boundary collides.
        self.assertFalse(np.array_equal(shares.probe(SEED,'p',8),shares.probe(SEED,'p',16)[:8]))
        self.assertFalse(np.array_equal(shares.probe(SEED,'ab',16),shares.probe(SEED,'a',16)))
        self.assertTrue(bool((shares.probe(SEED,'p',4096)<np.uint64(P)).all()))
        feeds=[np.array([P,0,5,P],dtype='<u8').tobytes(),np.array([7,9,11],dtype='<u8').tobytes()]
        drawn=shares._random((4,),lambda count:feeds.pop(0),nonzero=True)
        self.assertEqual(drawn.tolist(),[7,9,5,11])
        self.assertEqual(int(shares.subtract(np.uint64(0),np.uint64(P+1))),P-1)

    def test_a_caught_evaluator_that_then_leaves_is_still_reported_as_caught(self):
        class Leaves(Local):
            def receive(self,rows,width):
                if len(self.views)>1:raise ConnectionError('gone')
                answer=self.pending.copy();answer[0,0]=shares.add(answer[0,0],np.uint64(1));return answer
        route=shares.ShareRoute((Leaves(self.weights),Local(self.weights)),SEED,self.anchors)
        self.assertFalse(route.linear('p',self.inputs,20,12).any())
        with self.assertRaises(ExecutionRejected) as caught:route.linear('p',self.inputs,20,12)
        self.assertEqual(caught.exception.code,'integrity-check-failed')
        class Gone(Local):
            def receive(self,rows,width):raise ConnectionError('gone')
        route=shares.ShareRoute((Gone(self.weights),Local(self.weights)),SEED,self.anchors)
        with self.assertRaises(ExecutionRejected) as caught:route.linear('p',self.inputs,20,12)
        self.assertEqual(caught.exception.code,'execution-failed')

    def test_admission_and_strength(self):
        route,_,_=self.route()
        with self.assertRaises(ValueError):route.linear('q',self.inputs,20,12)
        with self.assertRaises(ValueError):route.linear('p',self.inputs.astype(np.int64),20,12)
        with self.assertRaises(ValueError):route.linear('p',np.full((1,48),4096,dtype=np.uint64),20,12)
        with self.assertRaises(ValueError):route.linear('p',np.zeros((shares.POLICY['maximumRows']+1,48),dtype=np.uint64),20,12)
        with self.assertRaises(ValueError):shares.product_bound(1<<21,12)
        bound,_,_=self.route(geometry={'p':(48,20)});bound.linear('p',self.inputs,20,12)
        wrong_width,_,_=self.route(geometry={'p':(64,20)})
        with self.assertRaises(ValueError):wrong_width.linear('p',self.inputs,20,12)
        late,_,_=self.route();late.linear('p',self.inputs,20,12);late.began-=shares.POLICY['attemptTimeoutSeconds']+1
        with self.assertRaises(ExecutionRejected) as caught:late.linear('p',self.inputs,20,12)
        self.assertEqual(caught.exception.code,'deadline-exceeded')
        with self.assertRaises(ValueError):shares.field_product(self.inputs,self.weights['p'].astype(np.int64))
        self.assertEqual(shares.detection_log2(2),120)
        for change in (dict(checkRounds=0),dict(fieldBits=64),dict(rejection='immediate'),dict(operationTimeoutSeconds=0)):
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

    def process_route(self,command,anchors=None,**options):
        return shares.ShareRoute((shares.ProcessEvaluator(command,'a',**options),shares.ProcessEvaluator(command,'b',**options)),SEED,anchors or self.anchors)

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

    def flight_directory(self,directory,changed=None):
        """Four projections, one large enough that a flight's requests and answers overflow any pipe buffer."""
        generator=np.random.default_rng(11)
        shapes={'q':(48,20),'k':(48,12),'v':(48,20),'big':(512,4096)}
        weights={name:generator.integers(-2048,2048,size=shape,dtype=np.int16) for name,shape in shapes.items()}
        served={name:matrix.copy() for name,matrix in weights.items()}
        if changed:served[changed][3,5]+=1
        rows=[]
        for name,matrix in served.items():
            data=matrix.astype('<i2').tobytes();identity=hashlib.sha256(data).hexdigest()
            (Path(directory)/(identity+'.i16')).write_bytes(data)
            rows.append(dict(name=name,inputWidth=matrix.shape[0],outputWidth=matrix.shape[1],weightId=identity,file=identity+'.i16',dtype='<i2'))
        (Path(directory)/'shares-manifest.json').write_text(json.dumps(dict(schema='doppler.two-server-shares-manifest/v1',projections=rows)))
        anchors={name:shares.build_anchor(SEED,name,matrix) for name,matrix in weights.items()}
        command=[sys.executable,str(Path(shares.__file__).with_name('shares_evaluator.py')),'--cache-dir',str(directory)]
        return weights,anchors,command

    def flight_calls(self,weights,rows=3):
        generator=np.random.default_rng(13)
        calls=[(name,generator.integers(0,4096,size=(rows if name!='big' else 24,matrix.shape[0]),dtype=np.uint64),matrix.shape[1],12)
               for name,matrix in weights.items()]
        return calls,[call[1].astype(np.int64)@weights[call[0]].astype(np.int64) for call in calls]

    def test_a_flight_returns_what_separate_calls_return_and_large_payloads_do_not_stall_it(self):
        with tempfile.TemporaryDirectory() as directory:
            weights,anchors,command=self.flight_directory(directory)
            calls,expected=self.flight_calls(weights)
            route=self.process_route(command,anchors)
            try:
                began=time.monotonic()
                for got,want in zip(route.linear_many(calls),expected):np.testing.assert_array_equal(got,want)
                self.assertLess(time.monotonic()-began,30)
                # The stream stays in step: a later single call and a second flight still line up.
                np.testing.assert_array_equal(route.linear(*calls[0]),expected[0])
                for got,want in zip(route.linear_many(calls[:3]),expected[:3]):np.testing.assert_array_equal(got,want)
                route.finish()
            finally:
                route.close()
            route.require_integrity()
            report=route.report()
            self.assertEqual((report['sharedProjectionCalls'],report['sharedFlights']),(8,3))
            self.assertEqual(report['sentBytes'][0],report['sentBytes'][1])
            # The same rows cross the wire whether or not they share a flight.
            separate=self.process_route(command,anchors)
            try:
                for call in calls+[calls[0]]+calls[:3]:separate.linear(*call)
            finally:
                separate.close()
            self.assertEqual(separate.report()['sentBytes'],report['sentBytes'])
            self.assertEqual(separate.report()['receivedBytes'],report['receivedBytes'])
            self.assertEqual(separate.report()['sharedFlights'],8)

    def test_a_flight_without_process_evaluators_and_an_empty_flight(self):
        route,a,_=self.route()
        got=route.linear_many([('p',self.inputs,20,12),('p',self.inputs[:1],20,12)])
        np.testing.assert_array_equal(got[0],self.expected);np.testing.assert_array_equal(got[1],self.expected[:1])
        self.assertEqual(len(a.views),2)
        with self.assertRaises(ValueError):route.linear_many([])
        with self.assertRaises(ValueError):route.linear_many([('p',self.inputs,20,12),('q',self.inputs,20,12)])

    def test_one_member_of_a_flight_failing_its_check_is_zeroed_and_the_rest_stand(self):
        with tempfile.TemporaryDirectory() as honest,tempfile.TemporaryDirectory() as dishonest:
            weights,anchors,command=self.flight_directory(honest)
            _,_,other=self.flight_directory(dishonest,changed='k')
            calls,expected=self.flight_calls(weights)
            route=shares.ShareRoute((shares.ProcessEvaluator(command,'a'),shares.ProcessEvaluator(other,'b')),SEED,anchors)
            try:
                got=route.linear_many(calls)
            finally:
                route.close()
            for (name,_,_,_),result,want in zip(calls,got,expected):
                if name=='k':self.assertFalse(result.any())
                else:np.testing.assert_array_equal(result,want)
            self.assertTrue(route.integrity_failed())
            with self.assertRaises(ExecutionRejected) as caught:route.require_integrity()
            self.assertEqual(caught.exception.code,'integrity-check-failed')

    def flight_quitter(self,zeros):
        """Answers the first request (honestly from the cache, or with zeros) and then exits mid-flight."""
        here=str(Path(shares.__file__).parent)
        return ('import sys,json,numpy as np\nsys.path.insert(0,%r)\nimport shares\nfrom pathlib import Path\n'
                'd=Path(sys.argv[1]);m={r["name"]:r for r in json.loads((d/"shares-manifest.json").read_text())["projections"]}\n'
                'r=sys.stdin.buffer;w=sys.stdout.buffer\n'
                'h=r.read(shares.REQUEST_HEAD.size);_,seq,n,rows,width=shares.REQUEST_HEAD.unpack(h);name=r.read(n).decode()\n'
                'x=np.frombuffer(r.read(8*rows*width),dtype="<u8").reshape(rows,width);p=m[name]\n'
                'W=np.memmap(d/p["file"],dtype=p["dtype"],mode="r",shape=(p["inputWidth"],p["outputWidth"]))\n'
                'a=shares.field_product(x,W)\n'
                'if %r:a=np.zeros_like(a)\n'
                'w.write(shares.RESPONSE_HEAD.pack(shares.RESPONSE,seq,0,a.shape[0],a.shape[1])+a.astype("<u8").tobytes());w.flush()\n')%(here,zeros)

    def test_a_provider_that_leaves_mid_flight_ends_the_attempt_and_one_already_caught_stays_caught(self):
        with tempfile.TemporaryDirectory() as directory:
            weights,anchors,command=self.flight_directory(directory)
            calls,_=self.flight_calls(weights)
            for zeros,code in ((False,'execution-failed'),(True,'integrity-check-failed')):
                quitter=[sys.executable,'-c',self.flight_quitter(zeros),str(directory)]
                route=shares.ShareRoute((shares.ProcessEvaluator(command,'a'),shares.ProcessEvaluator(quitter,'b')),SEED,anchors)
                try:
                    began=time.monotonic()
                    with self.assertRaises(ExecutionRejected) as caught:route.linear_many(calls[:3])
                    self.assertEqual(caught.exception.code,code)
                    self.assertLess(time.monotonic()-began,30)
                    self.assertEqual(route.integrity_failed(),zeros)
                    with self.assertRaises(ExecutionRejected):route.linear(*calls[0])
                finally:
                    route.close()

    def test_flight_partners_are_same_block_projections_that_read_one_tensor(self):
        block=lambda index:[f'model.layers.{index}.self_attn.{leaf}' for leaf in ('q_proj','k_proj','v_proj','o_proj')]+[
            f'model.layers.{index}.mlp.{leaf}' for leaf in ('gate_proj','up_proj','down_proj')]
        names=block(0)+block(1)+['lm_head']
        partners=shares.flight_partners(names)
        self.assertEqual(partners,{
            'model.layers.0.self_attn.q_proj':['model.layers.0.self_attn.k_proj','model.layers.0.self_attn.v_proj'],
            'model.layers.0.mlp.gate_proj':['model.layers.0.mlp.up_proj'],
            'model.layers.1.self_attn.q_proj':['model.layers.1.self_attn.k_proj','model.layers.1.self_attn.v_proj'],
            'model.layers.1.mlp.gate_proj':['model.layers.1.mlp.up_proj']})
        # 7 projections a block become 4 flights; the head stays alone. Nothing that consumes another's output is grouped.
        grouped={name for found in partners.values() for name in found}
        self.assertEqual(len(names)-len(grouped),2*4+1)
        self.assertFalse(any(name.endswith(('o_proj','down_proj','lm_head')) for name in list(partners)+list(grouped)))
        # A missing partner or a different parent is not grouped.
        self.assertEqual(shares.flight_partners(['a.q_proj','b.k_proj','a.v_proj']),{'a.q_proj':['a.v_proj']})
        self.assertEqual(shares.flight_partners(['q_proj','k_proj']),{'q_proj':['k_proj']})

    def test_evaluator_serves_only_regular_files_inside_its_directory(self):
        with tempfile.TemporaryDirectory() as outer:
            inner=Path(outer)/'cache';inner.mkdir()
            identity,_=self.provider_directory(outer)
            _,escaping=self.provider_directory(inner,file='../'+identity+'.i16')
            (inner/'link.i16').symlink_to(inner/(identity+'.i16'))
            _,linked=self.provider_directory(inner,file='link.i16')
            for command in (escaping,linked):
                route=self.process_route(command)
                try:
                    with self.assertRaises(ExecutionRejected):route.linear('p',self.inputs,20,12)
                finally:
                    route.close()
            self.provider_directory(inner)

    def test_a_caught_provider_process_that_then_exits_is_still_reported_as_caught(self):
        here=str(Path(shares.__file__).parent)
        # Answers the first request with zeros of the right shape, then exits.
        script=('import sys\nsys.path.insert(0,%r)\nimport shares\nr=sys.stdin.buffer;w=sys.stdout.buffer\n'
                'h=r.read(shares.REQUEST_HEAD.size);_,seq,n,rows,width=shares.REQUEST_HEAD.unpack(h);r.read(n);r.read(8*rows*width)\n'
                'w.write(shares.RESPONSE_HEAD.pack(shares.RESPONSE,seq,0,rows,20)+bytes(8*rows*20));w.flush()\n')%here
        route=self.process_route([sys.executable,'-c',script])
        try:
            self.assertFalse(route.linear('p',self.inputs,20,12).any())
            self.assertTrue(route.integrity_failed())
            with self.assertRaises(ExecutionRejected) as caught:route.linear('p',self.inputs,20,12)
            self.assertEqual(caught.exception.code,'integrity-check-failed')
        finally:
            route.close()

    def test_watchdog_completion_serializes_with_expiry(self):
        evaluator=shares.ProcessEvaluator.__new__(shares.ProcessEvaluator)
        evaluator.timeout=30;evaluator.attempt_deadline=time.monotonic()-1
        evaluator.cancelled=None;evaluator.interruption=None;evaluator.failed=False
        entered=threading.Event();release=threading.Event();done=threading.Event();errors=[]
        def kill():entered.set();release.wait(2)
        evaluator._kill=kill
        finish=evaluator._deadline()
        self.assertTrue(entered.wait(1))
        def complete():
            try:finish()
            except ExecutionRejected as error:errors.append(error.code)
            finally:done.set()
        completion=threading.Thread(target=complete);completion.start()
        self.assertFalse(done.wait(0.05))
        release.set();completion.join(2)
        self.assertTrue(done.is_set())
        self.assertEqual(errors,['deadline-exceeded'])
        # Completing before expiry joins/disarms the worker, so later exchanges cannot be killed.
        evaluator.attempt_deadline=time.monotonic()+0.1;evaluator.interruption=None
        kills=[];evaluator._kill=lambda:kills.append(True)
        evaluator._deadline()()
        time.sleep(0.15)
        self.assertEqual(kills,[])

    def test_attempt_deadline_and_cancel_interrupt_inflight_send_and_read(self):
        commands=[
            [sys.executable,'-c','import time; time.sleep(60)'],
            [sys.executable,'-c','import sys; sys.stdin.buffer.read()'],
        ]
        for command in commands:
            for cancel in (False,True):
                # Wide data blocks send; a small request fits the pipe and blocks receive.
                inputs=np.zeros((64,4096) if command is commands[0] else (1,48),dtype=np.uint64)
                route=self.process_route(command,anchors={'w':np.zeros(4,dtype=np.uint64)},timeout=30)
                route.began=time.monotonic()-shares.POLICY['attemptTimeoutSeconds']+0.3 if not cancel else time.monotonic()
                event=threading.Event();route.cancelled=event
                timer=threading.Timer(0.3,event.set) if cancel else None
                if timer:timer.start()
                began=time.monotonic()
                try:
                    with self.assertRaises(ExecutionRejected) as caught:route.linear('w',inputs,4,12)
                    self.assertEqual(caught.exception.code,'cancelled' if cancel else 'deadline-exceeded')
                    self.assertLess(time.monotonic()-began,3)
                    self.assertTrue(all(e.process.poll() is not None for e in (route.a,route.b)))
                finally:
                    if timer:timer.cancel()
                    route.close()

    def test_closing_and_deadlines_are_bounded_even_when_a_helper_keeps_the_pipes(self):
        wide=np.zeros((64,4096),dtype=np.uint64);anchors={'w':np.zeros(4,dtype=np.uint64)}
        # Reads most of the request so the pipe stays full, closes its output and sleeps.
        fills_then_sleeps=[sys.executable,'-c','import os,sys,time\nsys.stdin.buffer.read(6000000)\nos.close(1)\nwhile True:time.sleep(60)']
        # Leaves a helper process holding the pipes after it exits itself.
        leaves_helper=[sys.executable,'-c','import subprocess,sys\nsubprocess.Popen([sys.executable,"-c","import time\\nwhile True:time.sleep(60)"])\nsys.exit(0)']
        for command in (fills_then_sleeps,leaves_helper):
            route=self.process_route(command,anchors=anchors,timeout=1)
            began=time.monotonic()
            try:
                with self.assertRaises(ExecutionRejected):route.linear('w',wide,4,12)
            finally:
                route.close()
            self.assertLess(time.monotonic()-began,9,command[2][:40])

    def test_a_provider_that_stops_reading_or_never_answers_is_cut_off_at_the_deadline(self):
        wide=np.zeros((64,4096),dtype=np.uint64);anchors={'w':np.zeros(4,dtype=np.uint64)}
        never_reads=[sys.executable,'-c','import time\nwhile True:time.sleep(60)']
        never_answers=[sys.executable,'-c','import sys\nwhile True:sys.stdin.buffer.read(65536)']
        for command in (never_reads,never_answers):
            route=self.process_route(command,anchors=anchors,timeout=1)
            began=time.monotonic()
            try:
                with self.assertRaises(ExecutionRejected) as caught:route.linear('w',wide,4,12)
                self.assertEqual(caught.exception.code,'execution-failed')
                self.assertLess(time.monotonic()-began,8)
            finally:
                route.close()


if __name__=='__main__':
    unittest.main()
