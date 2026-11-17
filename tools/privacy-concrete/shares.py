"""Two-server additive secret sharing for the linear layers.

The client splits each quantized activation row into two uniformly random
shares modulo the prime 2**61 - 1 and sends one to each of two evaluators.
Each evaluator multiplies its share by the public integer matrix. The client
adds the two answers and gets the exact product. Neither evaluator sees
anything but uniform random numbers unless the two collude. Arithmetic is in a
prime field so that a deviation cannot hide behind a zero divisor.
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
MECHANISM='additive-shares-m61-hidden-probe/v1'
PRIME=(1<<61)-1
REQUEST,RESPONSE,CLOSE=b'SHQ1',b'SHR1',b'SHC1'
REQUEST_HEAD,RESPONSE_HEAD=struct.Struct('<4sQHII'),struct.Struct('<4sQBII')


def validate_policy(policy):
    if (policy.get('schema')!='doppler.two-server-shares-policy/v1' or policy.get('mechanism')!=MECHANISM or policy.get('fieldBits')!=61
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
    if bound>=1<<59 or input_width>=1<<21:raise ValueError('Projection exceeds the share field')
    return bound


def detection_log2(rounds):
    """One evaluator that changes an output value must guess a field element per round: at most 2**-this."""
    return 60*rounds


COMMON_DEVIATION_LOG2=60  # Two evaluators using the same other matrix must cancel on one unseen probe row.


def _reduce(values):
    """Values below 2**63 to their residue below the prime."""
    import numpy as np
    prime=np.uint64(PRIME)
    values=(values&prime)+(values>>np.uint64(61))
    return values-prime*(values>=prime).astype(np.uint64)


def _shift(values,bits):
    """values * 2**bits modulo the prime, for residues below it: a 61-bit rotation."""
    import numpy as np
    low=(values&np.uint64((1<<(61-bits))-1))<<np.uint64(bits)
    return _reduce(low+(values>>np.uint64(61-bits)))


def add(left,right):
    return _reduce(left+right)


def subtract(left,right):
    import numpy as np
    return _reduce(left+(np.uint64(PRIME)-right))


def multiply(left,right):
    """Elementwise product modulo the prime, through 31-bit halves so nothing leaves 64 bits."""
    import numpy as np
    mask=np.uint64((1<<31)-1);half=np.uint64(31)
    left_low,left_high,right_low,right_high=left&mask,left>>half,right&mask,right>>half
    middle=_reduce(left_high*right_low)+_reduce(left_low*right_high)
    return _reduce(_reduce(left_low*right_low)+_shift(_reduce(middle),31)+_reduce((left_high*right_high)<<np.uint64(1)))


def _field(data,shape):
    """Bytes to field elements. The top residue folds onto zero, a bias of 2**-61."""
    import numpy as np
    values=np.frombuffer(data,dtype='<u8').reshape(shape)&np.uint64(PRIME)
    return np.where(values==np.uint64(PRIME),np.uint64(0),values)


def _random(shape,rng,nonzero=False):
    import numpy as np
    values=_field(rng(8*int(np.prod(shape))),shape)
    return np.where(values==0,np.uint64(1),values) if nonzero else values


def probe(seed,name,width):
    """The requester's secret probe row for a projection. Providers never see it or anything that reveals it."""
    if not isinstance(seed,bytes) or len(seed)<32:raise ValueError('Secret probe seed of at least 32 bytes required')
    return _field(hashlib.shake_256(b'doppler-shares-probe/v2\0'+seed+b'\0'+name.encode()).digest(8*width),(width,))


def field_product(rows,weights,chunk_elements=POLICY['maximumChunkElements']):
    """rows (R, n) of field elements times weights (n, m) of small signed integers, modulo the prime.

    Each element is split into four limbs of at most 16 bits. A limb times a 16-bit weight is below 2**31 and a sum
    over fewer than 2**21 inputs stays below 2**53, so the float64 matrix product is exact.
    """
    import numpy as np
    if (rows.dtype!=np.uint64 or rows.ndim!=2 or weights.ndim!=2 or rows.shape[1]!=weights.shape[0] or weights.shape[0]>=1<<21
            or weights.dtype.kind!='i' or weights.dtype.itemsize>2):
        raise ValueError('Explicit share geometry required')
    count=rows.shape[0]
    limbs=np.concatenate([((rows>>np.uint64(16*k))&np.uint64(0xFFFF)).astype(np.float64) for k in range(4)])
    output=np.empty((count,weights.shape[1]),dtype=np.uint64)
    step=max(1,chunk_elements//weights.shape[0])
    for begin in range(0,weights.shape[1],step):
        block=np.asarray(weights[:,begin:begin+step],dtype=np.float64)
        partial=np.mod((limbs@block).astype(np.int64),np.int64(PRIME)).astype(np.uint64)
        total=partial[:count]
        for k in range(1,4):total=add(total,_shift(partial[k*count:(k+1)*count],16*k))
        output[:,begin:begin+step]=total
    return output


def build_anchor(seed,name,weights):
    """The probe's exact product with the true matrix. Computed once by the requester, or by someone it trusts."""
    return field_product(probe(seed,name,weights.shape[0]).reshape(1,-1),weights)[0]


def build_queries(inputs,probe_row,rounds,rng=os.urandom):
    """Rows for evaluator A and B, and the secrets the client keeps to check their answers.

    A receives x1, then per round alpha*x1 + delta*probe + t, then per round u.
    B receives x2, then per round beta*x2 + epsilon*probe + u, then per round t.
    t and u are fresh uniform masks, so each evaluator's rows are jointly uniform whatever the input and probe are.
    """
    import numpy as np
    rows,width=inputs.shape
    first=_random((rows,width),rng)
    second=subtract(inputs,first)
    alpha,beta=_random((rounds,rows,1),rng),_random((rounds,rows,1),rng)
    delta,epsilon=_random((rounds,rows,1),rng,nonzero=True),_random((rounds,rows,1),rng,nonzero=True)
    t,u=_random((rounds,rows,width),rng),_random((rounds,rows,width),rng)
    hidden=probe_row.reshape(1,-1)
    check=lambda multiplier,share,scale,mask:add(add(multiply(multiplier,share),multiply(scale,hidden)),mask)
    a=np.concatenate([first]+[check(alpha[k],first,delta[k],t[k]) for k in range(rounds)]+[u[k] for k in range(rounds)])
    b=np.concatenate([second]+[check(beta[k],second,epsilon[k],u[k]) for k in range(rounds)]+[t[k] for k in range(rounds)])
    return a,b,dict(alpha=alpha,beta=beta,delta=delta,epsilon=epsilon,rows=rows,rounds=rounds)


def combine(answer_a,answer_b,secrets,anchor,bound):
    """Exact signed product and whether every check passed. Nothing here depends on which check failed."""
    import numpy as np
    rows,rounds=secrets['rows'],secrets['rounds']
    expected=(1+2*rounds)*rows
    if (answer_a.shape!=answer_b.shape or answer_a.shape[0]!=expected or answer_a.dtype!=np.uint64 or answer_b.dtype!=np.uint64
            or bool((answer_a>=np.uint64(PRIME)).any()) or bool((answer_b>=np.uint64(PRIME)).any())):
        return None,False
    part=lambda answer,index:answer[index*rows:(index+1)*rows]
    target=anchor.reshape(1,-1)
    passed=True
    for k in range(rounds):
        # Each check ties one evaluator's answers to the other's and, through the hidden probe, to the true matrix.
        left=subtract(subtract(part(answer_a,1+k),multiply(secrets['alpha'][k],part(answer_a,0))),part(answer_b,1+rounds+k))
        passed&=bool(np.array_equal(left,multiply(secrets['delta'][k],target)))
        right=subtract(subtract(part(answer_b,1+k),multiply(secrets['beta'][k],part(answer_b,0))),part(answer_a,1+rounds+k))
        passed&=bool(np.array_equal(right,multiply(secrets['epsilon'][k],target)))
    total=add(part(answer_a,0),part(answer_b,0))
    product=np.where(total>np.uint64(PRIME//2),total.view(np.int64)-np.int64(PRIME),total.view(np.int64))
    # A sanity bound, not the integrity mechanism: a wrong value only reaches it after passing the field checks.
    passed&=bool(((product>-bound)&(product<bound)).all())
    return product,passed


class ProcessEvaluator:
    """One evaluator reached through a child process's pipes; the command may be local or an SSH invocation."""
    def __init__(self,command,label,timeout=POLICY['operationTimeoutSeconds']):
        self.label,self.sequence,self.sent,self.received,self.timeout=label,0,0,0,timeout
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)

    def _deadline(self):
        # A provider that stops reading or answering is cut off; the blocked call then ends as a closed stream.
        timer=threading.Timer(self.timeout,self.process.kill);timer.daemon=True;timer.start()
        return timer

    def send(self,name,rows):
        encoded=name.encode()
        payload=rows.astype('<u8',copy=False).tobytes()
        self.sequence+=1
        timer=self._deadline()
        try:
            self.process.stdin.write(REQUEST_HEAD.pack(REQUEST,self.sequence,len(encoded),rows.shape[0],rows.shape[1])+encoded)
            self.process.stdin.write(payload);self.process.stdin.flush()
        finally:
            timer.cancel()
        self.sent+=REQUEST_HEAD.size+len(encoded)+len(payload)

    def _read(self,count):
        data=self.process.stdout.read(count)
        if len(data)!=count:raise ConnectionError('Evaluator closed its stream')
        self.received+=count
        return data

    def receive(self,rows,width):
        import numpy as np
        timer=self._deadline()
        try:
            magic,sequence,status,got_rows,got_width=RESPONSE_HEAD.unpack(self._read(RESPONSE_HEAD.size))
            if magic!=RESPONSE or sequence!=self.sequence or status!=0 or got_rows!=rows or got_width!=width:
                raise ConnectionError('Evaluator answered out of contract')
            return np.frombuffer(self._read(8*rows*width),dtype='<u8').reshape(rows,width)
        finally:
            timer.cancel()

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

    def check_active(self):
        if self.closed:raise ExecutionRejected('contract-rejected')
        if self.cancelled is not None and self.cancelled.is_set():raise ExecutionRejected('cancelled')

    def linear(self,name,inputs,output_width,quantization_bits):
        """Exact inputs @ W for the named projection, or zeros after a recorded failure."""
        import numpy as np
        self.check_active()
        if (inputs.dtype!=np.uint64 or inputs.ndim!=2 or not 1<=inputs.shape[0]<=self.policy['maximumRows']
                or name not in self.anchors or self.anchors[name].shape!=(output_width,)
                or bool((inputs>=np.uint64(1<<quantization_bits)).any())):
            raise ValueError('Projection is outside the admitted share route')
        rounds=self.policy['checkRounds']
        bound=product_bound(inputs.shape[1],quantization_bits)
        began=time.perf_counter()
        query_a,query_b,secrets=build_queries(inputs,probe(self.probe_seed,name,inputs.shape[1]),rounds,self.rng)
        built=time.perf_counter()
        try:
            self.a.send(name,query_a);self.b.send(name,query_b)
            answer_a=self.a.receive(query_a.shape[0],output_width)
            answer_b=self.b.receive(query_b.shape[0],output_width)
        except (ConnectionError,OSError):
            # An evaluator that was already caught cannot turn that into a transport error by leaving.
            self.closed=True
            raise ExecutionRejected('integrity-check-failed' if self.failure is not None else 'execution-failed') from None
        answered=time.perf_counter()
        product,passed=combine(answer_a,answer_b,secrets,self.anchors[name],bound)
        finished=time.perf_counter()
        self.calls+=1;self.rows+=inputs.shape[0]
        self.client_seconds+=(built-began)+(finished-answered);self.remote_seconds+=answered-built
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
                    singleEvaluatorDetectionLog2=detection_log2(self.policy['checkRounds']),commonDeviationDetectionLog2=COMMON_DEVIATION_LOG2,
                    clientShareMs=self.client_seconds*1000,evaluatorRoundTripMs=self.remote_seconds*1000,
                    sentBytes=[self.a.sent,self.b.sent],receivedBytes=[self.a.received,self.b.received])
