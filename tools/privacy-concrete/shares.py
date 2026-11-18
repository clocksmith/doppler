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
import signal
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
            or type(policy.get('operationTimeoutSeconds')) is not int or not 1<=policy['operationTimeoutSeconds']<=3600
            or type(policy.get('attemptTimeoutSeconds')) is not int or not 1<=policy['attemptTimeoutSeconds']<=86400):
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
    """Sum of two residues below the prime."""
    return _reduce(left+right)


def subtract(left,right):
    """Difference of two residues; the subtrahend is reduced first so an unreduced value cannot underflow."""
    import numpy as np
    return _reduce(left+(np.uint64(PRIME)-_reduce(right)))


def multiply(left,right):
    """Elementwise product modulo the prime, through 31-bit halves so nothing leaves 64 bits."""
    import numpy as np
    mask=np.uint64((1<<31)-1);half=np.uint64(31)
    left_low,left_high,right_low,right_high=left&mask,left>>half,right&mask,right>>half
    middle=_reduce(left_high*right_low)+_reduce(left_low*right_high)
    return _reduce(_reduce(left_low*right_low)+_shift(_reduce(middle),31)+_reduce((left_high*right_high)<<np.uint64(1)))


def _field(data,shape):
    """Bytes to candidate field elements: 61 bits each. The one value equal to the prime is not a residue."""
    import numpy as np
    return np.frombuffer(data,dtype='<u8').reshape(shape)&np.uint64(PRIME)


def _random(shape,rng,nonzero=False):
    """Exactly uniform field elements: a draw equal to the prime (or zero, when excluded) is drawn again."""
    import numpy as np
    values=_field(rng(8*int(np.prod(shape))),shape).copy()
    while True:
        bad=(values==np.uint64(PRIME))|((values==0) if nonzero else False)
        count=int(bad.sum())
        if not count:return values
        values[bad]=_field(rng(8*count),(count,))


def probe(seed,name,width):
    """The requester's secret probe row for a projection. Providers never see it or anything that reveals it."""
    import numpy as np
    if not isinstance(seed,bytes) or len(seed)!=32 or type(width) is not int or not 0<width<1<<21:
        raise ValueError('A 32-byte secret probe seed and an explicit width are required')
    encoded=name.encode()
    stream=hashlib.shake_256(b'doppler-shares-probe/v3\0'+seed+width.to_bytes(4,'little')+len(encoded).to_bytes(4,'little')+encoded).digest(8*(width+16))
    values=_field(stream,(width+16,))
    values=values[values!=np.uint64(PRIME)][:width]
    if values.shape[0]!=width:raise ValueError('Probe derivation failed')
    return values.copy()


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
        self.label,self.sequence,self.sent,self.received,self.timeout,self.failed=label,0,0,0,timeout,False
        self.attempt_deadline=self.cancelled=None
        self.interruption=None
        # Its own process group, so a deadline also ends any helper that inherited the pipes.
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,start_new_session=True)

    def _kill(self):
        try:os.killpg(self.process.pid,signal.SIGKILL)
        except (ProcessLookupError,PermissionError):pass

    def bind_attempt(self,deadline,cancelled):
        self.attempt_deadline,self.cancelled=deadline,cancelled

    def _deadline(self,seconds=None):
        # One absolute attempt deadline spans all operations, including a blocked pipe.
        # Cleanup uses its own bounded grace period and ignores the cancelled attempt.
        stop=threading.Event();lock=threading.Lock()
        end=time.monotonic()+(self.timeout if seconds is None else seconds)
        if seconds is None and self.attempt_deadline is not None:end=min(end,self.attempt_deadline)
        def expire():
            now=time.monotonic()
            cancelled=seconds is None and self.cancelled is not None and self.cancelled.is_set()
            if cancelled or now>=end:
                self.interruption=('cancelled' if cancelled else
                    'deadline-exceeded' if seconds is None and self.attempt_deadline is not None and now>=self.attempt_deadline else 'execution-failed')
                self.failed=True;self._kill();return True
            return False
        def watch():
            while True:
                with lock:
                    if stop.is_set() or expire():return
                stop.wait(max(0,min(0.02,end-time.monotonic())))
        worker=threading.Thread(target=watch,daemon=True);worker.start()
        def finish():
            # Serialize completion with expiry: no watchdog can kill a later exchange.
            with lock:
                if not stop.is_set() and self.interruption is None:expire()
                stop.set()
            worker.join()
            if seconds is None and self.interruption is not None:
                raise ExecutionRejected(self.interruption)
        return finish

    def send(self,name,rows):
        encoded=name.encode()
        payload=rows.astype('<u8',copy=False).tobytes()
        self.sequence+=1
        timer=self._deadline()
        try:
            self.process.stdin.write(REQUEST_HEAD.pack(REQUEST,self.sequence,len(encoded),rows.shape[0],rows.shape[1])+encoded)
            self.process.stdin.write(payload);self.process.stdin.flush()
        except BaseException:
            self.failed=True;raise
        finally:
            timer()
        self.sent+=REQUEST_HEAD.size+len(encoded)+len(payload)
        return self.sequence

    def _read(self,count):
        data=self.process.stdout.read(count)
        if len(data)!=count:raise ConnectionError('Evaluator closed its stream')
        self.received+=count
        return data

    def receive(self,rows,width,expected=None):
        # `expected` names the request this answer must belong to when several requests are in flight;
        # without it the answer must belong to the request sent last.
        import numpy as np
        timer=self._deadline()
        try:
            magic,sequence,status,got_rows,got_width=RESPONSE_HEAD.unpack(self._read(RESPONSE_HEAD.size))
            if magic!=RESPONSE or sequence!=(self.sequence if expected is None else expected) or status!=0 or got_rows!=rows or got_width!=width:
                raise ConnectionError('Evaluator answered out of contract')
            return np.frombuffer(self._read(8*rows*width),dtype='<u8').reshape(rows,width)
        except BaseException:
            self.failed=True;raise
        finally:
            timer()

    def close(self):
        # After any failed exchange the provider is not asked to stop: it is ended. A polite close is itself bounded.
        if self.failed:self._kill()
        timer=self._deadline(5)
        try:
            if self.process.poll() is None:
                self.process.stdin.write(CLOSE);self.process.stdin.flush();self.process.stdin.close()
                self.process.wait(timeout=5)
        except (OSError,ValueError,subprocess.TimeoutExpired):
            self._kill()
        finally:
            timer()
            self._kill()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
            for stream in (self.process.stdin,self.process.stdout):
                try:stream.close()
                except OSError:pass


class ShareRoute:
    """Client side of one attempt, under the installed policy.

    Holds no weight matrix: its secret probe seed, one anchor row per projection and its own randomness. A failed
    check is recorded and answered with zeros; the caller must call require_integrity() after close() and release
    nothing if it raises. `geometry`, when given, fixes each projection's input and output width.
    """
    def __init__(self,evaluators,probe_seed,anchors,*,geometry=None,cancelled=None,rng=os.urandom,observe=lambda kind,value:None):
        if len(evaluators)!=2:raise ValueError('Exactly two evaluators required')
        self.policy=POLICY
        self.a,self.b=evaluators
        self.probe_seed,self.anchors,self.geometry,self.cancelled,self.rng,self.observe=probe_seed,anchors,geometry,cancelled,rng,observe
        self.began=None
        self.calls=self.rows=self.flights=0
        self.failure=None
        self.closed=False
        self.client_seconds=self.remote_seconds=0.0

    def check_active(self):
        if self.closed:raise ExecutionRejected('contract-rejected')
        if self.cancelled is not None and self.cancelled.is_set():raise ExecutionRejected('cancelled')
        if self.began is not None and time.monotonic()-self.began>self.policy['attemptTimeoutSeconds']:
            self.closed=True;raise ExecutionRejected('integrity-check-failed' if self.failure is not None else 'deadline-exceeded')

    def linear(self,name,inputs,output_width,quantization_bits):
        """Exact inputs @ W for the named projection, or zeros after a recorded failure."""
        import numpy as np
        self.check_active()
        if (inputs.dtype!=np.uint64 or inputs.ndim!=2 or not 1<=inputs.shape[0]<=self.policy['maximumRows']
                or name not in self.anchors or self.anchors[name].shape!=(output_width,)
                or (self.geometry is not None and self.geometry.get(name)!=(inputs.shape[1],output_width))
                or bool((inputs>=np.uint64(1<<quantization_bits)).any())):
            raise ValueError('Projection is outside the admitted share route')
        if self.began is None:self.began=time.monotonic()
        deadline=self.began+self.policy['attemptTimeoutSeconds']
        for evaluator in (self.a,self.b):
            if isinstance(evaluator,ProcessEvaluator):evaluator.bind_attempt(deadline,self.cancelled)
        rounds=self.policy['checkRounds']
        bound=product_bound(inputs.shape[1],quantization_bits)
        began=time.perf_counter()
        query_a,query_b,secrets=build_queries(inputs,probe(self.probe_seed,name,inputs.shape[1]),rounds,self.rng)
        built=time.perf_counter()
        try:
            self.check_active();self.a.send(name,query_a)
            self.check_active();self.b.send(name,query_b)
            self.check_active();answer_a=self.a.receive(query_a.shape[0],output_width)
            self.check_active();answer_b=self.b.receive(query_b.shape[0],output_width)
            self.check_active()
        except (ConnectionError,OSError,ExecutionRejected) as error:
            # An evaluator that was already caught cannot turn that into a transport error by leaving.
            reason='execution-failed'
            if isinstance(error,ExecutionRejected):reason=error.code
            elif self.cancelled is not None and self.cancelled.is_set():reason='cancelled'
            elif time.monotonic()>=deadline:reason='deadline-exceeded'
            for evaluator in (self.a,self.b):
                if isinstance(evaluator,ProcessEvaluator):evaluator.failed=True
            self.close()
            raise ExecutionRejected('integrity-check-failed' if self.failure is not None else reason) from None
        answered=time.perf_counter()
        product,passed=combine(answer_a,answer_b,secrets,self.anchors[name],bound)
        finished=time.perf_counter()
        self.calls+=1;self.rows+=inputs.shape[0];self.flights+=1
        self.client_seconds+=(built-began)+(finished-answered);self.remote_seconds+=answered-built
        self.observe('shared-projection',dict(name=name,rows=int(inputs.shape[0])))
        if not passed:
            self.failure=self.failure or 'check'
            return np.zeros((inputs.shape[0],output_width),dtype=np.int64)
        return product

    def linear_many(self,calls):
        """Several independent projections in one flight: [(name, inputs, output_width, quantization_bits), ...].

        Returns one result per call, in order, each exactly what linear() would have returned. Every query is
        built and checked as in linear(); the only difference is that all of a flight's queries are sent before
        any answer is awaited, so a flight costs one network round trip per provider and not one per projection.
        The providers see the same rows in the same order. Only projections whose inputs do not depend on one
        another's outputs may share a flight (query, key and value; the two inputs of an MLP gate).
        """
        import numpy as np
        calls=list(calls)
        if not calls:raise ValueError('A flight needs at least one projection')
        if len(calls)==1 or not all(isinstance(evaluator,ProcessEvaluator) for evaluator in (self.a,self.b)):
            return [self.linear(*call) for call in calls]
        self.check_active()
        for name,inputs,output_width,quantization_bits in calls:
            if (inputs.dtype!=np.uint64 or inputs.ndim!=2 or not 1<=inputs.shape[0]<=self.policy['maximumRows']
                    or name not in self.anchors or self.anchors[name].shape!=(output_width,)
                    or (self.geometry is not None and self.geometry.get(name)!=(inputs.shape[1],output_width))
                    or bool((inputs>=np.uint64(1<<quantization_bits)).any())):
                raise ValueError('Projection is outside the admitted share route')
        if self.began is None:self.began=time.monotonic()
        deadline=self.began+self.policy['attemptTimeoutSeconds']
        for evaluator in (self.a,self.b):evaluator.bind_attempt(deadline,self.cancelled)
        rounds=self.policy['checkRounds']
        began=time.perf_counter()
        built=[build_queries(inputs,probe(self.probe_seed,name,inputs.shape[1]),rounds,self.rng) for name,inputs,_,_ in calls]
        ready=time.perf_counter()

        def flight(evaluator,side,answers,errors):
            # A writer thread sends while this thread reads, so neither pipe can fill and stall the other.
            queries=[(call[0],pair[side],call[2]) for call,pair in zip(calls,built)]
            first=evaluator.sequence+1
            def write():
                try:
                    for name,rows,_ in queries:
                        self.check_active();evaluator.send(name,rows)
                except BaseException as error:errors.append(error)
            writer=threading.Thread(target=write,daemon=True);writer.start()
            try:
                for index,(_,rows,width) in enumerate(queries):
                    self.check_active();answers.append(evaluator.receive(rows.shape[0],width,first+index))
            except BaseException as error:
                errors.append(error);evaluator.failed=True;evaluator._kill()
            finally:
                writer.join()

        answers_a,answers_b,errors=[],[],[]
        other=threading.Thread(target=flight,args=(self.b,1,answers_b,errors),daemon=True);other.start()
        flight(self.a,0,answers_a,errors);other.join()
        def judge(index):
            name,inputs,output_width,quantization_bits=calls[index]
            return combine(answers_a[index],answers_b[index],built[index][2],self.anchors[name],product_bound(inputs.shape[1],quantization_bits))
        if errors:
            error=errors[0]
            if not isinstance(error,(ConnectionError,OSError,ExecutionRejected)):raise error
            # An evaluator that was already caught cannot turn that into a transport error by leaving: every
            # projection both providers did answer is still checked before the attempt is reported.
            for index in range(min(len(answers_a),len(answers_b))):
                if not judge(index)[1]:self.failure=self.failure or 'check'
            reason='execution-failed'
            rejected=[item for item in errors if isinstance(item,ExecutionRejected)]
            if rejected:reason=rejected[0].code
            elif self.cancelled is not None and self.cancelled.is_set():reason='cancelled'
            elif time.monotonic()>=deadline:reason='deadline-exceeded'
            for evaluator in (self.a,self.b):evaluator.failed=True
            self.close()
            raise ExecutionRejected('integrity-check-failed' if self.failure is not None else reason) from None
        answered=time.perf_counter()
        results=[]
        for index,(name,inputs,output_width,quantization_bits) in enumerate(calls):
            product,passed=judge(index)
            self.calls+=1;self.rows+=inputs.shape[0]
            self.observe('shared-projection',dict(name=name,rows=int(inputs.shape[0])))
            if not passed:
                self.failure=self.failure or 'check'
                product=np.zeros((inputs.shape[0],output_width),dtype=np.int64)
            results.append(product)
        finished=time.perf_counter()
        self.flights+=1
        self.client_seconds+=(ready-began)+(finished-answered);self.remote_seconds+=answered-ready
        return results

    def integrity_failed(self):return self.failure is not None

    def require_integrity(self):
        if self.failure is not None:raise ExecutionRejected('integrity-check-failed')

    def finish(self):self.check_active()

    def close(self):
        self.closed=True
        for evaluator in (self.a,self.b):evaluator.close()

    def report(self):
        return dict(integrity=MECHANISM,integrityPolicyId=POLICY_ID,checkRounds=self.policy['checkRounds'],
                    sharedProjectionCalls=self.calls,sharedRows=self.rows,sharedFlights=self.flights,
                    singleEvaluatorDetectionLog2=detection_log2(self.policy['checkRounds']),commonDeviationDetectionLog2=COMMON_DEVIATION_LOG2,
                    clientShareMs=self.client_seconds*1000,evaluatorRoundTripMs=self.remote_seconds*1000,
                    sentBytes=[self.a.sent,self.b.sent],receivedBytes=[self.a.received,self.b.received])


# Projections of one block that read the same tensor and do not feed one another, so they may share a flight.
FLIGHT_GROUPS=(('q_proj','k_proj','v_proj'),('gate_proj','up_proj'))


def flight_partners(names):
    """Map each flight's first projection to the partners that travel with it: {lead name: [partner names]}.

    Names are dotted module paths. Two projections are partners only when they sit under the same parent module
    and their leaf names belong to one FLIGHT_GROUPS entry; the lead is the entry's first leaf, which is the one
    the model calls first.
    """
    present=set(names);partners={}
    for name in names:
        parent,_,leaf=name.rpartition('.')
        for group in FLIGHT_GROUPS:
            if leaf==group[0]:
                found=[(parent+'.' if parent else '')+other for other in group[1:] if (parent+'.' if parent else '')+other in present]
                if found:partners[name]=found
    return partners


def transport_label(command,verified_network=False):
    exe=Path(command[0]).name
    if exe=='ssh':return 'ssh'
    if verified_network and any(Path(part).name=='shares_network.js' for part in command[1:3]) and 'connect' in command:
        return 'shares-net-ed25519-chacha20poly1305/v1'
    return 'local-process'


def parse_evaluator_commands(raw_items):
    if raw_items is None:return None
    commands=[json.loads(item) for item in raw_items]
    if len(commands)!=2 or not all(isinstance(cmd,list) and cmd and all(isinstance(part,str) and part for part in cmd) for cmd in commands):
        raise ValueError('Exactly two non-empty string-list evaluator commands required')
    return commands


def build_network_commands(enrollment_file,provider_endpoints,plan_id,model_id,manifest_sha256,requester_owner=None,node_binary='node'):
    if len(provider_endpoints)!=2:raise ValueError('Exactly two --provider-endpoint entries (<computerId>@<host>:<port>) required')
    enrollment_path=Path(enrollment_file)
    if enrollment_path.is_symlink() or not enrollment_path.is_file():
        raise ValueError('Regular non-symlink enrollment file required')
    enrollment_sha256=hashlib.sha256(enrollment_path.read_bytes()).hexdigest()
    parsed=[]
    seen_computers=set()
    seen_endpoints=set()
    for item in provider_endpoints:
        if '@' not in item:raise ValueError(f'Expected <computerId>@<host>:<port>, got {item}')
        computer_id,endpoint=item.split('@',1)
        if not computer_id or computer_id.startswith('--') or ':' not in endpoint or endpoint.startswith('--'):
            raise ValueError(f'Invalid provider endpoint {item}')
        norm_endpoint=endpoint.strip().lower()
        if computer_id in seen_computers:raise ValueError(f'Duplicate provider computerId: {computer_id}')
        if norm_endpoint in seen_endpoints:raise ValueError(f'Duplicate provider endpoint: {endpoint}')
        seen_computers.add(computer_id);seen_endpoints.add(norm_endpoint)
        parsed.append((computer_id,endpoint.strip()))
    if requester_owner is not None and (not requester_owner.strip() or requester_owner.startswith('--')):
        raise ValueError('Invalid --requester-owner')
    script=str(Path(__file__).with_name('shares_network.js'))
    verify_cmd=[node_binary,script,'verify-enrollment','--enrollment',str(enrollment_path),
                '--enrollment-sha256',enrollment_sha256,
                '--provider-a',parsed[0][0],'--provider-b',parsed[1][0]]
    if requester_owner is not None:verify_cmd.extend(['--requester-owner',requester_owner])
    checked=subprocess.run(verify_cmd,capture_output=True,text=True,timeout=15,check=False)
    if checked.returncode!=0:
        raise ValueError(checked.stderr.strip() or 'Provider enrollment verification failed')
    verification=json.loads(checked.stdout)
    peer_ids=[parsed[1][0],parsed[0][0]]
    commands=[]
    for (computer_id,endpoint),peer_id,role in zip(parsed,peer_ids,'ab'):
        cmd=[node_binary,script,'connect','--endpoint',endpoint,'--enrollment',str(enrollment_path),
             '--enrollment-sha256',enrollment_sha256,'--computer-id',computer_id,'--peer-computer-id',peer_id,
             '--role',role,'--plan-id',plan_id,'--model-id',model_id,
             '--policy-id',POLICY_ID,'--manifest-sha256',manifest_sha256]
        if requester_owner is not None:cmd.extend(['--requester-owner',requester_owner])
        commands.append(cmd)
    return commands,verification
