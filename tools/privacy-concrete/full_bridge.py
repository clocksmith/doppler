"""One immutable all-layer route; verify each exact encrypted tile before client decryption."""
import hmac
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import concrete_ml_extensions as ext
from protocol import check, digest, receive, require_active, send
from full_plan import load_plan
from full_admission import FullRouteAdmission
from full_status import ExecutionRejected
from public_weights import bind_public_tiles


class FullBridge:
    def __init__(self, directory, plan, key, cancelled=None, observe=lambda kind, value: None, deployment=None, backend_profile=None):
        require_active(cancelled)
        if backend_profile is None:validated=load_plan(directory)
        else:
            from plan_backends import load_backend_plan
            validated=load_backend_plan(directory,backend_profile)
        if validated.identity!=plan.identity:
            raise ValueError('Client full-model admission differs from the prepared plan')
        bound=validated if deployment is None else deployment.bind(validated)
        self.admission=FullRouteAdmission(bound)
        self.public_tile_digests=bind_public_tiles(directory,validated)
        self.plan, self.key, self.cancelled, self.observe = bound, key, cancelled, observe
        self.payload_limit=self.plan.definition()['parameters'].get('maxPayloadBytes')
        self.check_active()
        self.session, data = uuid.uuid4().hex, key.serialize()
        self.check_active()
        self.key_id, self.key_bytes = digest(data), len(data)
        self.sequence, self.sent, self.received = 0, 0, 0
        self.remote_seconds = self.replay_seconds = 0.0
        self.child = deployment.spawn(validated) if deployment is not None else subprocess.Popen([sys.executable,str(Path(__file__).with_name('full_provider.py')),str(Path(directory).resolve()),plan.identity,*(['--profile',backend_profile] if backend_profile is not None else [])],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0,
                                      env={'RAYON_NUM_THREADS':str(self.plan.definition()['cpuThreads']),'OMP_NUM_THREADS':str(self.plan.definition()['cpuThreads'])})
        try:
            self.exchange('evaluation-key', data, 'ready')
        except BaseException:
            self.close()
            raise

    def check_active(self):
        try:
            require_active(self.cancelled)
        except BaseException:
            self.admission.fail()
            raise

    def exchange(self, kind, payload, response_kind, tile_id=''):
        header = self.plan.frame(self.session,self.key_id,self.sequence,kind,tile_id)
        self.sent += send(self.child.stdin,header,payload,timeout=self.plan.definition()['operationTimeoutSeconds'],cancelled=self.cancelled,max_payload_bytes=self.payload_limit)
        reply, result, size = receive(self.child.stdout,timeout=self.plan.definition()['operationTimeoutSeconds'],cancelled=self.cancelled,max_payload_bytes=self.payload_limit)
        self.received += size
        check(reply,self.plan.frame(self.session,self.key_id,self.sequence,response_kind,tile_id))
        self.sequence += 1
        return result

    def evaluate(self, encrypted, matrix, tile):
        try:
            self.check_active()
            expected_tile=self.admission.admit_tile(tile,matrix.shape,self.sequence)
            if matrix.dtype!=np.uint64:
                raise ValueError('Encrypted route requires unsigned native matrix representation')
            signed=matrix.view(np.int64)
            owned=np.ascontiguousarray(signed,dtype=np.int8)
            if not np.array_equal(signed,owned) or digest(owned.tobytes())!=self.public_tile_digests[expected_tile['id']]:
                raise ValueError('Encrypted route public weight changed before disclosure')
            start = time.perf_counter()
            result = self.exchange('evaluate',encrypted.serialize(),'result',tile['id'])
            self.remote_seconds += time.perf_counter()-start
            self.check_active()
            start = time.perf_counter()
            expected = ext.cpu_matrix_multiplication(encrypted,matrix,self.key).serialize()
            self.replay_seconds += time.perf_counter()-start
            self.check_active()
            if not hmac.compare_digest(expected,result):
                raise ExecutionRejected('ciphertext-verification-failed')
            answer=ext.CompressedResultEncryptedMatrix.deserialize(result)
            self.admission.verified()
            self.observe('verified-tile',dict(sequence=self.sequence-1,projection=tile['name'],tileId=tile['id']))
            return answer
        except BaseException:
            self.admission.fail()
            raise

    def finish(self):
        self.check_active()
        self.admission.complete()
        count = len(self.plan.schedule())*self.plan.definition()['steps']
        if self.sequence != count+1:
            raise ValueError('Incomplete full encrypted route')
        self.sent += send(self.child.stdin,self.plan.frame(self.session,self.key_id,self.sequence,'finish'),timeout=self.plan.definition()['operationTimeoutSeconds'],cancelled=self.cancelled,max_payload_bytes=self.payload_limit)
        self.child.stdin.close()
        status=self.child.wait(timeout=180)
        self.check_active()
        if status != 0 or self.child.stdout.read(1):
            raise ValueError('Full evaluator failed to finish')

    def close(self):
        if self.child.poll() is None:
            self.child.kill()
        self.child.wait()
        for stream in (self.child.stdin,self.child.stdout):
            if stream and not stream.closed:
                stream.close()
