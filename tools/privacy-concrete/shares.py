"""Two-server additive secret sharing for the linear layers.

The client splits each quantized activation row into two uniformly random
shares over the integers modulo 2**64 and sends one to each of two evaluators.
Each evaluator multiplies its share by the public integer matrix at ordinary
speed. The client adds the two answers and gets the exact product. Neither
evaluator sees anything but uniform random numbers unless the two collude.
See SHARES.md for the argument and its limits.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import threading
import time

from full_status import ExecutionRejected

_FILE=Path(__file__).with_name('shares-policy.json')
POLICY=json.loads(_FILE.read_text())
POLICY_ID=hashlib.sha256(_FILE.read_bytes()).hexdigest()
MECHANISM='additive-shares-z2e64-hidden-probe/v1'
REQUEST,RESPONSE,CLOSE=b'SHQ1',b'SHR1',b'SHC1'
REQUEST_HEAD,RESPONSE_HEAD=struct.Struct('<4sQHII'),struct.Struct('<4sQBII')


def validate_policy(policy):
    if (policy.get('schema')!='doppler.two-server-shares-policy/v1' or policy.get('mechanism')!=MECHANISM or policy.get('ringBits')!=64
            or type(policy.get('checkRounds')) is not int or not 1<=policy['checkRounds']<=4
            or policy.get('rejection')!='deferred-until-route-close/v1'
            or type(policy.get('maximumRows')) is not int or not 1<=policy['maximumRows']<=4096
            or type(policy.get('maximumChunkElements')) is not int or policy['maximumChunkElements']<=0
            or type(policy.get('operationTimeoutSeconds')) is not int or not 1<=policy['operationTimeoutSeconds']<=3600):
        raise ValueError('Unsupported two-server shares policy')
    return policy


validate_policy(POLICY)


def product_bound(input_width,quantization_bits):
    """Largest magnitude an honest product can reach: inputs in [0, 2**q), weights in [-2**(q-1), 2**(q-1))."""
    bound=input_width*(1<<quantization_bits)*(1<<(quantization_bits-1))
    if bound>=1<<62:raise ValueError('Projection exceeds the share ring')
    return bound


def detection_log2(bound,rounds):
    """An evaluator that changes one output value escapes with probability at most 2**-this."""
    return rounds*(64-(2*bound-1).bit_length())


def probe(seed,name,width):
    """The requester's secret probe row for a projection. Providers never see it or anything that reveals it."""
    import numpy as np
    if not isinstance(seed,bytes) or len(seed)<32:raise ValueError('Secret probe seed of at least 32 bytes required')
    stream=hashlib.shake_256(b'doppler-shares-probe/v1\0'+seed+b'\0'+name.encode()).digest(8*width)
    return np.frombuffer(stream,dtype='<u8').copy()


def ring_product(rows,weights,chunk_elements=POLICY['maximumChunkElements']):
    """rows (R, n) uint64 times weights (n, m) signed integers, modulo 2**64, in bounded column chunks."""
    import numpy as np
    if rows.dtype!=np.uint64 or rows.ndim!=2 or weights.ndim!=2 or rows.shape[1]!=weights.shape[0]:
        raise ValueError('Explicit share geometry required')
    output=np.empty((rows.shape[0],weights.shape[1]),dtype=np.uint64)
    step=max(1,chunk_elements//weights.shape[0])
    for begin in range(0,weights.shape[1],step):
        block=np.asarray(weights[:,begin:begin+step]).astype(np.int64).view(np.uint64)
        output[:,begin:begin+step]=rows@block
    return output


def build_anchor(seed,name,weights):
    """The probe's exact product with the true matrix. Computed once by the requester, or by someone it trusts."""
    return ring_product(probe(seed,name,weights.shape[0]).reshape(1,-1),weights)[0]


def _random(shape,rng):
    import numpy as np
    count=int(np.prod(shape))
    return np.frombuffer(rng(8*count),dtype='<u8').reshape(shape).copy()


def build_queries(inputs,probe_row,rounds,rng=os.urandom):
    """Rows for evaluator A and B, and the secrets the client keeps to check their answers.

    A receives x1, then per round alpha*x1 + delta*probe + t, then per round u.
    B receives x2, then per round beta*x2 + epsilon*probe + u, then per round t.
    t and u are fresh uniform masks, so each evaluator's rows are jointly uniform and no row is recognisable.
    """
    import numpy as np
    rows,width=inputs.shape
    first=_random((rows,width),rng)
    second=inputs-first
    alpha,beta=_random((rounds,rows,1),rng),_random((rounds,rows,1),rng)
    delta,epsilon=_random((rounds,rows,1),rng)|np.uint64(1),_random((rounds,rows,1),rng)|np.uint64(1)
    t,u=_random((rounds,rows,width),rng),_random((rounds,rows,width),rng)
    hidden=probe_row.reshape(1,-1)
    a=np.concatenate([first]+[alpha[k]*first+delta[k]*hidden+t[k] for k in range(rounds)]+[u[k] for k in range(rounds)])
    b=np.concatenate([second]+[beta[k]*second+epsilon[k]*hidden+u[k] for k in range(rounds)]+[t[k] for k in range(rounds)])
    return a,b,dict(alpha=alpha,beta=beta,delta=delta,epsilon=epsilon,rows=rows,rounds=rounds)


def combine(answer_a,answer_b,secrets,anchor,bound):
    """Exact signed product and whether every check passed. Nothing here depends on which check failed."""
    import numpy as np
    rows,rounds=secrets['rows'],secrets['rounds']
    expected=(1+2*rounds)*rows
    if answer_a.shape!=answer_b.shape or answer_a.shape[0]!=expected or answer_a.dtype!=np.uint64 or answer_b.dtype!=np.uint64:
        return None,False
    part=lambda answer,index:answer[index*rows:(index+1)*rows]
    target=anchor.reshape(1,-1)
    passed=True
    for k in range(rounds):
        # Each check ties one evaluator's answers to the other's and, through the hidden probe, to the true matrix.
        passed&=bool(np.array_equal(part(answer_a,1+k)-secrets['alpha'][k]*part(answer_a,0)-part(answer_b,1+rounds+k),secrets['delta'][k]*target))
        passed&=bool(np.array_equal(part(answer_b,1+k)-secrets['beta'][k]*part(answer_b,0)-part(answer_a,1+rounds+k),secrets['epsilon'][k]*target))
    product=(part(answer_a,0)+part(answer_b,0)).view(np.int64)
    passed&=bool(((product>-bound)&(product<bound)).all())
    return product,passed


class ProcessEvaluator:
    """One evaluator reached through a child process's pipes; the command may be local or an SSH invocation."""
    def __init__(self,command,label,timeout=POLICY['operationTimeoutSeconds']):
        self.label,self.sequence,self.sent,self.received,self.timeout=label,0,0,0,timeout
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)

    def send(self,name,rows):
        encoded=name.encode()
        payload=rows.astype('<u8',copy=False).tobytes()
        self.sequence+=1
        self.process.stdin.write(REQUEST_HEAD.pack(REQUEST,self.sequence,len(encoded),rows.shape[0],rows.shape[1])+encoded)
        self.process.stdin.write(payload);self.process.stdin.flush()
        self.sent+=REQUEST_HEAD.size+len(encoded)+len(payload)

    def _read(self,count):
        data=self.process.stdout.read(count)
        if len(data)!=count:raise ConnectionError('Evaluator closed its stream')
        self.received+=count
        return data

    def receive(self,rows,width):
        import numpy as np
        # A provider that stalls is cut off at the deadline; the blocked read then ends as a closed stream.
        deadline=threading.Timer(self.timeout,self.process.kill);deadline.daemon=True;deadline.start()
        try:
            magic,sequence,status,got_rows,got_width=RESPONSE_HEAD.unpack(self._read(RESPONSE_HEAD.size))
            if magic!=RESPONSE or sequence!=self.sequence or status!=0 or got_rows!=rows or got_width!=width:
                raise ConnectionError('Evaluator answered out of contract')
            return np.frombuffer(self._read(8*rows*width),dtype='<u8').reshape(rows,width)
        finally:
            deadline.cancel()

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write(CLOSE);self.process.stdin.flush();self.process.stdin.close()
                self.process.wait(timeout=5)
        except (OSError,ValueError,subprocess.TimeoutExpired):
            self.process.kill();self.process.wait()
        finally:
            for stream in (self.process.stdin,self.process.stdout):
                try:stream.close()
                except OSError:pass


class ShareRoute:
    """Client side of one attempt. Holds no weight matrix: its secret probe seed, one anchor row per projection and its own randomness."""
    def __init__(self,evaluators,probe_seed,anchors,*,policy=POLICY,cancelled=None,rng=os.urandom,observe=lambda kind,value:None):
        if len(evaluators)!=2:raise ValueError('Exactly two evaluators required')
        self.policy=validate_policy(policy)
        self.a,self.b=evaluators
        self.probe_seed,self.anchors,self.cancelled,self.rng,self.observe=probe_seed,anchors,cancelled,rng,observe
        self.calls=self.rows=0
        self.failure=None
        self.closed=False
        self.client_seconds=self.remote_seconds=0.0
        self.minimum_detection_log2=None

    def check_active(self):
        if self.closed:raise ExecutionRejected('contract-rejected')
        if self.cancelled is not None and self.cancelled.is_set():raise ExecutionRejected('cancelled')

    def linear(self,name,inputs,output_width,quantization_bits):
        """Exact inputs @ W for the named projection, or zeros after a recorded failure."""
        import numpy as np
        self.check_active()
        if (inputs.dtype!=np.uint64 or inputs.ndim!=2 or not 1<=inputs.shape[0]<=self.policy['maximumRows']
                or name not in self.anchors or self.anchors[name].shape!=(output_width,)):
            raise ValueError('Projection is outside the admitted share route')
        rounds=self.policy['checkRounds']
        bound=product_bound(inputs.shape[1],quantization_bits)
        began=time.perf_counter()
        query_a,query_b,secrets=build_queries(inputs,probe(self.probe_seed,name,inputs.shape[1]),rounds,self.rng)
        built=time.perf_counter()
        product,passed=None,False
        try:
            self.a.send(name,query_a);self.b.send(name,query_b)
            answer_a=self.a.receive(query_a.shape[0],output_width)
            answer_b=self.b.receive(query_b.shape[0],output_width)
            answered=time.perf_counter()
            product,passed=combine(answer_a,answer_b,secrets,self.anchors[name],bound)
        except (ConnectionError,OSError):
            # An evaluator that was already caught cannot turn that into a transport error by leaving.
            self.closed=True
            raise ExecutionRejected('integrity-check-failed' if self.failure=='check' else 'execution-failed') from None
        finished=time.perf_counter()
        self.calls+=1;self.rows+=inputs.shape[0]
        self.client_seconds+=(built-began)+(finished-answered);self.remote_seconds+=answered-built
        strength=detection_log2(bound,rounds)
        self.minimum_detection_log2=strength if self.minimum_detection_log2 is None else min(self.minimum_detection_log2,strength)
        self.observe('shared-projection',dict(name=name,rows=int(inputs.shape[0])))
        if not passed:
            self.failure=self.failure or 'check'
            return np.zeros((inputs.shape[0],output_width),dtype=np.int64)
        return product

    def integrity_failed(self):return self.failure is not None

    def require_integrity(self):
        if self.failure is not None:raise ExecutionRejected('integrity-check-failed')

    def finish(self):self.check_active()

    def close(self):
        self.closed=True
        for evaluator in (self.a,self.b):evaluator.close()

    def report(self):
        return dict(integrity=MECHANISM,integrityPolicyId=POLICY_ID,checkRounds=self.policy['checkRounds'],
                    sharedProjectionCalls=self.calls,sharedRows=self.rows,
                    minimumDetectionLog2=self.minimum_detection_log2,
                    clientShareMs=self.client_seconds*1000,evaluatorRoundTripMs=self.remote_seconds*1000,
                    sentBytes=[self.a.sent,self.b.sent],receivedBytes=[self.a.received,self.b.received])
