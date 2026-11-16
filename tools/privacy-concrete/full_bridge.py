"""One immutable all-layer route; verify each exact encrypted tile before client decryption."""
import hmac
from pathlib import Path
import subprocess
import sys
import time
import uuid
import concrete_ml_extensions as ext
from protocol import check, digest, receive, send
from full_plan import FULL_POLICY


class FullBridge:
    def __init__(self, directory, plan, key, cancelled=None, observe=lambda kind, value: None):
        self.plan, self.key, self.cancelled, self.observe = plan, key, cancelled, observe
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
        start = time.perf_counter()
        result = self.exchange('evaluate',encrypted.serialize(),'result',tile['id'])
        self.remote_seconds += time.perf_counter()-start
        start = time.perf_counter()
        expected = ext.cpu_matrix_multiplication(encrypted,matrix,self.key).serialize()
        self.replay_seconds += time.perf_counter()-start
        if not hmac.compare_digest(expected,result):
            raise ValueError('Full encrypted tile rejected before decryption')
        self.observe('verified-tile',dict(sequence=self.sequence-1,projection=tile['name'],tileId=tile['id']))
        return ext.CompressedResultEncryptedMatrix.deserialize(result)

    def finish(self):
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
