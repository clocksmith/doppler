"""One immutable all-layer route; verify each exact encrypted tile before client decryption."""
import hmac
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import concrete_ml_extensions as ext
from protocol import check, digest, receive, send
from full_plan import FULL_POLICY, load_plan
from full_admission import FullRouteAdmission


class FullBridge:
    def __init__(self, directory, plan, key, cancelled=None, observe=lambda kind, value: None):
        validated=load_plan(directory)
        if validated.identity!=plan.identity:
            raise ValueError('Client full-model admission differs from the prepared plan')
        self.admission=FullRouteAdmission(validated)
        self.public_weights={row['name']:np.memmap(Path(directory)/(row['weightId']+'.i8'),dtype=np.int8,mode='r',shape=(row['inputWidth'],row['outputWidth'])) for row in validated.definition()['projections']}
        self.plan, self.key, self.cancelled, self.observe = validated, key, cancelled, observe
        self.session, data = uuid.uuid4().hex, key.serialize()
        self.key_id, self.key_bytes = digest(data), len(data)
        self.sequence, self.sent, self.received = 0, 0, 0
        self.remote_seconds = self.replay_seconds = 0.0
        self.child = subprocess.Popen([sys.executable,str(Path(__file__).with_name('full_provider.py')),str(Path(directory).resolve()),plan.identity],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0,
                                      env={'RAYON_NUM_THREADS':str(FULL_POLICY['cpuThreads']),'OMP_NUM_THREADS':str(FULL_POLICY['cpuThreads'])})
        try:
            self.exchange('evaluation-key', data, 'ready')
        except BaseException:
            self.close()
            raise

    def exchange(self, kind, payload, response_kind, tile_id=''):
        header = self.plan.frame(self.session,self.key_id,self.sequence,kind,tile_id)
        self.sent += send(self.child.stdin,header,payload,timeout=self.plan.definition()['operationTimeoutSeconds'],cancelled=self.cancelled)
        reply, result, size = receive(self.child.stdout,timeout=self.plan.definition()['operationTimeoutSeconds'],cancelled=self.cancelled)
        self.received += size
        check(reply,self.plan.frame(self.session,self.key_id,self.sequence,response_kind,tile_id))
        self.sequence += 1
        return result

    def evaluate(self, encrypted, matrix, tile):
        try:
            expected_tile=self.admission.admit_tile(tile,matrix.shape,self.sequence)
            cached=self.public_weights[expected_tile['name']][expected_tile['inputBegin']:expected_tile['inputEnd'],expected_tile['outputBegin']:expected_tile['outputEnd']]
            if matrix.dtype!=np.uint64 or not np.array_equal(matrix.view(np.int64),cached):
                raise ValueError('Encrypted route public weight changed before disclosure')
            start = time.perf_counter()
            result = self.exchange('evaluate',encrypted.serialize(),'result',tile['id'])
            self.remote_seconds += time.perf_counter()-start
            start = time.perf_counter()
            expected = ext.cpu_matrix_multiplication(encrypted,matrix,self.key).serialize()
            self.replay_seconds += time.perf_counter()-start
            if not hmac.compare_digest(expected,result):
                raise ValueError('Full encrypted tile rejected before decryption')
            answer=ext.CompressedResultEncryptedMatrix.deserialize(result)
            self.admission.verified()
            self.observe('verified-tile',dict(sequence=self.sequence-1,projection=tile['name'],tileId=tile['id']))
            return answer
        except BaseException:
            self.admission.fail()
            raise

    def finish(self):
        self.admission.complete()
        count = len(self.plan.schedule())*self.plan.definition()['steps']
        if self.sequence != count+1:
            raise ValueError('Incomplete full encrypted route')
        self.sent += send(self.child.stdin,self.plan.frame(self.session,self.key_id,self.sequence,'finish'),timeout=self.plan.definition()['operationTimeoutSeconds'],cancelled=self.cancelled)
        self.child.stdin.close()
        if self.child.wait(timeout=180) != 0 or self.child.stdout.read(1):
            raise ValueError('Full evaluator failed to finish')

    def close(self):
        if self.child.poll() is None:
            self.child.kill()
        self.child.wait()
        for stream in (self.child.stdin,self.child.stdout):
            if stream and not stream.closed:
                stream.close()
