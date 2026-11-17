"""All-layer route whose integrity is a secret linear check plus sampled exact replay.

Differences from FullBridge, each with a consequence stated in INTEGRITY.md:
the client decrypts responses it has not byte-verified; rejection is recorded
client-locally and raised only after the fixed route has finished and closed;
a rejected response is replaced by zeros so later inputs do not depend on it.
Research only: no independent cryptographic review, no production admission.
"""
import hmac
import os
import time
import numpy as np
import concrete_ml_extensions as ext
from full_bridge import FullBridge
from full_status import ExecutionRejected
from linear_check import CheckKey,POLICY,tolerance
from check_policy import require_bound_policy
from protocol import digest

INTEGRITY_ID=POLICY['id']
CHECKED_PROTOCOL_SUFFIX='-linear-check/v1'
PASS_THROUGH=(KeyboardInterrupt,SystemExit,InterruptedError,MemoryError,TimeoutError)


class CheckedBridge(FullBridge):
    def __init__(self,*args,parameter_profile,**kwargs):
        plan=kwargs.get('plan') if 'plan' in kwargs else args[1] if len(args)>1 else None
        if plan is None or not plan.protocol_id.endswith(CHECKED_PROTOCOL_SUFFIX):
            raise ValueError('Plan protocol does not declare checked integrity')
        # Pure admission before the base class can spawn or disclose a key.
        require_bound_policy(plan,parameter_profile)
        super().__init__(*args,**kwargs)
        try:self.begin_checks(parameter_profile)
        except BaseException:
            self.close()
            raise

    def begin_checks(self,parameter_profile,seed=None):
        # A replay-protocol plan must never be executed with the weaker check.
        if not self.plan.protocol_id.endswith(CHECKED_PROTOCOL_SUFFIX):raise ValueError('Plan protocol does not declare checked integrity')
        if getattr(self,'check',None) is not None:raise ValueError('Attempt check policy and secret cannot be replaced')
        snapshot=require_bound_policy(self.plan,parameter_profile)
        definition=self.plan.definition()
        self.check=CheckKey(os.urandom(32) if seed is None else seed,parameter_profile,
            definition['parameters']['parameters']['bits_reserved_for_computation'],definition['parameters']['quantizationBits'],policy=snapshot)
        self.failures=[];self.result_lengths={};self._pending=None
        self.linear_checks=self.sampled_replays=self.substituted_tiles=0
        self.check_seconds=0.0

    def _fail(self,kind):
        # Client-local only. Nothing about a failure is sent or changes the route.
        self.failures.append(dict(kind=kind,sequence=self.sequence-1))

    def evaluate(self,encrypted,matrix,tile):
        try:
            self.check_active()
            if self._pending is not None:raise ValueError('Previous checked response was not decoded')
            schedule=len(self.admission.schedule);step=self.admission.position//schedule
            rows=self.admission.prefill_rows if step==0 else 1
            expected_tile=self.admission.admit_tile(tile,matrix.shape,self.sequence)
            if matrix.dtype!=np.uint64:
                raise ValueError('Encrypted route requires unsigned native matrix representation')
            signed=matrix.view(np.int64)
            owned=np.ascontiguousarray(signed,dtype=self.weight_dtype)
            if not np.array_equal(signed,owned) or digest(owned.tobytes())!=self.public_tile_digests[expected_tile['id']]:
                raise ValueError('Encrypted route public weight changed before disclosure')
            start=time.perf_counter()
            expectation=self.check.expectation(expected_tile['id'],owned,step,rows)
            self.check_seconds+=time.perf_counter()-start
            geometry=(rows,expected_tile['outputEnd']-expected_tile['outputBegin'])
            # The first response of each geometry is always replayed: its trusted
            # local length then bounds what later responses may hand the parser.
            sampled=geometry not in self.result_lengths or self.check.sampled(expected_tile['id'],step)
            start=time.perf_counter()
            try:
                # Never replay while the response is pending: a client that stops
                # reading would reveal the sample before the evaluator has sent
                # the whole response. The complete reply is in memory first.
                result=self.exchange('evaluate',self._serialize_input(encrypted,expected_tile),'result',tile['id'])
            finally:
                self.remote_seconds+=time.perf_counter()-start
            self.check_active()
            accepted=True
            if sampled:
                began=time.perf_counter()
                try:expected=self.matrix_multiplication(encrypted,matrix,self.key).serialize()
                finally:self.replay_seconds+=time.perf_counter()-began
                self.check_active()
                self.sampled_replays+=1
                self.result_lengths.setdefault(geometry,len(expected))
                if not hmac.compare_digest(expected,result):
                    accepted=False;self._fail('sampled-replay-mismatch')
            elif len(result)!=self.result_lengths[geometry]:
                accepted=False;self._fail('response-length')
            answer=None
            if accepted:
                try:answer=ext.CompressedResultEncryptedMatrix.deserialize(result)
                except PASS_THROUGH:raise
                except BaseException:
                    self._fail('response-parse')
            self._pending=(expected_tile['id'],expectation,rows,step)
            self.admission.verified()
            self.observe('received-tile',dict(sequence=self.sequence-1,projection=tile['name'],tileId=tile['id']))
            return answer
        except BaseException:
            self.admission.fail()
            self._release_projection_inputs()
            raise

    def decode(self,answer,tile,inputs,decrypt):
        try:
            self.check_active()
            pending,self._pending=self._pending,None
            if pending is None or pending[0]!=tile['id']:raise ValueError('Checked response does not belong to this tile')
            _,expectation,rows,step=pending
            outputs=tile['outputEnd']-tile['outputBegin']
            values=np.ascontiguousarray(inputs).view(np.int64)
            if values.shape!=(rows,tile['inputEnd']-tile['inputBegin']):raise ValueError('Checked input geometry changed')
            substitute=np.zeros((rows,outputs),dtype=np.int64)
            if answer is None:
                self.substituted_tiles+=1;return substitute
            start=time.perf_counter()
            try:
                try:decoded=np.ascontiguousarray(decrypt(answer),dtype=np.int64)
                except PASS_THROUGH:raise
                except BaseException:
                    self._fail('response-decrypt');self.substituted_tiles+=1;return substitute
                if decoded.shape!=substitute.shape or not self.check.passes(tile['id'],values,decoded,expectation,step):
                    self._fail('linear-check');self.substituted_tiles+=1;return substitute
                self.linear_checks+=1
                return decoded
            finally:self.check_seconds+=time.perf_counter()-start
        except BaseException:
            self.admission.fail()
            self._release_projection_inputs()
            raise

    def integrity_failed(self):
        """True if any response was rejected. Valid on every exit path, closed or not."""
        return bool(self.failures) or self._pending is not None

    def require_integrity(self):
        """Call only after the route has finished and the transport is closed."""
        if self.child.poll() is None or not self.check.retired:raise ValueError('Integrity outcome requires a closed route')
        if self.integrity_failed():
            self.admission.fail()
            raise ExecutionRejected('integrity-check-failed')

    def close(self):
        try:super().close()
        finally:
            check=getattr(self,'check',None)
            if check is not None:check.retire()

    def integrity_report(self):
        widths=sorted({width for _,width in self.result_lengths})
        return dict(integrity=self.check.policy['id'],integrityPolicyId=self.check.policy_id,linearChecksPassed=self.linear_checks,sampledExactReplays=self.sampled_replays,
            substitutedTiles=self.substituted_tiles,clientCheckMs=self.check_seconds*1000,checkVectors=self.check.count,
            replaySample=dict(self.check.policy['replaySample']),toleranceByOutputWidth={str(width):tolerance(width,self.check.noise,self.check.policy) for width in widths},
            rejection=self.check.policy['rejection'],independentCryptographicReview=False,securityQualified=False)
