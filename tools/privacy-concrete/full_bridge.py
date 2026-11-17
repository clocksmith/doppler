"""One immutable all-layer route; verify each exact encrypted tile before client decryption."""
import hmac
from contextlib import contextmanager
from functools import cached_property
from types import MappingProxyType
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
from weight_encoding import weight_encoding
from native_runtime import matrix_evaluator,prepare_evaluator_key


class FullBridge:
    def __init__(self, directory, plan, key, cancelled=None, observe=lambda kind, value: None, deployment=None, backend_profile=None,prepared_public_tiles=None,provider_source_directory=None):
        require_active(cancelled)
        from plan_backends import STREAMED_PROFILE,REBUILD_PROFILES,compiled_backend
        self.streams_public_coefficients=backend_profile==STREAMED_PROFILE
        self.rebuilds_public_coefficients=backend_profile in REBUILD_PROFILES
        compiled=self.streams_public_coefficients or self.rebuilds_public_coefficients
        self.public_coefficient_bytes_sent=0
        if provider_source_directory is not None and (not self.rebuilds_public_coefficients or deployment is not None):
            raise ValueError('Explicit local public checkpoint applies only to the local rebuild backend')
        if self.rebuilds_public_coefficients and deployment is None:
            if provider_source_directory is None or not Path(provider_source_directory).is_dir():
                raise ValueError('Existing local provider checkpoint required; no source download or fallback')
        if compiled:
            from checked_bridge import CheckedBridge
            if type(self) is not CheckedBridge:raise ValueError('Compiled public profile requires its declared checked-integrity bridge')
            validated=compiled_backend(backend_profile).load_client_plan(directory,prepared_public_tiles)
        elif prepared_public_tiles is not None:raise ValueError('Compiled tile binding requires an explicit streamed backend')
        elif backend_profile is None:validated=load_plan(directory)
        else:
            from plan_backends import load_backend_plan
            validated=load_backend_plan(directory,backend_profile)
        if validated.identity!=plan.identity:
            raise ValueError('Client full-model admission differs from the prepared plan')
        bound=validated if deployment is None else deployment.bind(validated)
        self.admission=FullRouteAdmission(bound)
        self.public_tile_digests=(prepared_public_tiles.digests_for(validated) if compiled else bind_public_tiles(directory,validated))
        self.plan, self.key, self.cancelled, self.observe = bound, key, cancelled, observe
        self.weight_encoding=weight_encoding(self.plan.definition()['parameters'])
        self.weight_dtype=self.weight_encoding['dtype']
        self.payload_limit=self.plan.definition()['parameters'].get('maxPayloadBytes')
        self.check_active()
        self.matrix_multiplication=matrix_evaluator(self.plan)
        prepare_evaluator_key(self.plan,key)
        self.check_active()
        self.session, data = uuid.uuid4().hex, key.serialize()
        self.check_active()
        self.key_id, self.key_bytes = digest(data), len(data)
        self.sequence, self.sent, self.received = 0, 0, 0
        self.remote_seconds = self.replay_seconds = 0.0
        self.child = deployment.spawn(validated) if deployment is not None else subprocess.Popen([sys.executable,str(Path(__file__).with_name('full_provider.py')),str(Path(directory).resolve()),plan.identity,*(['--profile',backend_profile] if backend_profile is not None else []),*(['--public-source-dir',str(Path(provider_source_directory).resolve())] if self.rebuilds_public_coefficients else [])],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0,
                                      env={'RAYON_NUM_THREADS':str(self.plan.definition()['cpuThreads']),'OMP_NUM_THREADS':str(self.plan.definition()['cpuThreads'])})
        try:
            self.exchange('evaluation-key', data, 'ready')
        except BaseException:
            self.close()
            raise

    @cached_property
    def operation_timeout(self):return self.plan.definition()['operationTimeoutSeconds']

    @cached_property
    def execution_geometry(self):
        definition=self.plan.definition()
        return MappingProxyType(dict(inputTileWidth=definition['inputTileWidth'],outputTileWidth=definition['outputTileWidth'],
            arithmeticBits=definition['parameters']['parameters']['bits_reserved_for_computation']))

    def check_active(self):
        try:
            require_active(self.cancelled)
            if self.admission.failed:raise ValueError('Encrypted attempt is terminally unavailable')
        except BaseException:
            self.admission.fail()
            self._release_projection_inputs()
            raise

    def _release_projection_inputs(self):
        inputs=getattr(self,'_projection_inputs',None)
        if inputs is not None:inputs.clear()
        self._projection_inputs=None
        self._public_tile_payload=None

    def stage_public_tile(self,matrix,tile):
        """Own only the current public tile; identical bytes reach checks and wire."""
        try:
            self.check_active()
            if not self.streams_public_coefficients:raise ValueError('Public coefficient streaming is not admitted')
            expected=self.admission.next_tile()
            if dict(tile)!=dict(expected) or matrix.dtype!=np.uint64:raise ValueError('Unadmitted streamed public tile')
            signed=matrix.view(np.int64);owned=np.ascontiguousarray(signed,dtype=self.weight_dtype)
            payload=owned.tobytes()
            if (tuple(matrix.shape)!=(tile['inputEnd']-tile['inputBegin'],tile['outputEnd']-tile['outputBegin'])
                    or not np.array_equal(signed,owned) or digest(payload)!=self.public_tile_digests[tile['id']]):
                raise ValueError('Streamed public matrix differs from client preparation')
            self._public_tile_payload=(tile['id'],payload)
        except BaseException:
            self.admission.fail();self._release_projection_inputs();raise

    @contextmanager
    def projection(self,name,input_width,output_width,rows):
        """Own at most one serialized input for this admitted projection and key."""
        try:
            self.check_active()
            if getattr(self,'_projection_inputs',None) is not None:
                raise ValueError('Encrypted projection scope is already active')
            self.admission.admit_projection(name,input_width,output_width,rows)
            self._projection_inputs=[]
            yield
            if self.admission.failed or self.admission.projection is not None:
                raise ValueError('Incomplete encrypted projection scope')
        except BaseException:
            self.admission.fail()
            raise
        finally:
            self._release_projection_inputs()

    def _serialize_ciphertext(self,encrypted,tile):
        inputs=getattr(self,'_projection_inputs',None)
        geometry=(tile['inputBegin'],tile['inputEnd'])
        if inputs and inputs[0][0]==geometry and inputs[0][1] is encrypted and inputs[0][2] is self.key:
            return inputs[0][3]
        # Release the previous payload before allocating its replacement. In
        # particular, never retain every input tile's serialized ciphertext.
        if inputs is not None:inputs.clear()
        payload=encrypted.serialize()
        self.check_active()
        if inputs is not None:inputs.append((geometry,encrypted,self.key,payload))
        return payload

    def _serialize_input(self,encrypted,tile):
        payload=self._serialize_ciphertext(encrypted,tile)
        if not getattr(self,'streams_public_coefficients',False):return payload
        public=getattr(self,'_public_tile_payload',None)
        if public is None or public[0]!=tile['id']:raise ValueError('Current admitted public coefficient tile is unavailable')
        from public_tile_stream import encode
        return encode(public[1],payload,tile,self.weight_encoding,self.public_tile_digests[tile['id']])

    def exchange(self, kind, payload, response_kind, tile_id='', while_pending=None):
        # Sending first lets the evaluator compute while the client performs its
        # mandatory deterministic replay. There is still only one admitted tile.
        timeout=self.operation_timeout
        if kind=='evaluation-key' and getattr(self,'rebuilds_public_coefficients',False):
            timeout=self.plan.definition()['sourceVerificationTimeoutSeconds']
        deadline=time.monotonic()+timeout
        def remaining():
            self.check_active()
            value=deadline-time.monotonic()
            if value<=0:raise TimeoutError('Encrypted operation deadline exceeded')
            return value
        header = self.plan.frame(self.session,self.key_id,self.sequence,kind,tile_id)
        self.sent += send(self.child.stdin,header,payload,timeout=remaining(),cancelled=self.cancelled,max_payload_bytes=self.payload_limit)
        if kind=='evaluate' and getattr(self,'streams_public_coefficients',False):
            self.public_coefficient_bytes_sent+=len(self._public_tile_payload[1])
        if while_pending is not None:
            remaining()
            while_pending()
        reply, result, size = receive(self.child.stdout,timeout=remaining(),cancelled=self.cancelled,max_payload_bytes=self.payload_limit)
        self.received += size
        check(reply,self.plan.frame(self.session,self.key_id,self.sequence,response_kind,tile_id))
        remaining()
        self.sequence += 1
        return result

    def evaluate(self, encrypted, matrix, tile):
        try:
            self.check_active()
            expected_tile=self.admission.admit_tile(tile,matrix.shape,self.sequence)
            if matrix.dtype!=np.uint64:
                raise ValueError('Encrypted route requires unsigned native matrix representation')
            signed=matrix.view(np.int64)
            owned=np.ascontiguousarray(signed,dtype=self.weight_dtype)
            if not np.array_equal(signed,owned) or digest(owned.tobytes())!=self.public_tile_digests[expected_tile['id']]:
                raise ValueError('Encrypted route public weight changed before disclosure')
            expected=None
            def replay():
                nonlocal expected
                self.check_active()
                start=time.perf_counter()
                try:
                    expected=self.matrix_multiplication(encrypted,matrix,self.key).serialize()
                finally:
                    self.replay_seconds+=time.perf_counter()-start
                self.check_active()
            start=time.perf_counter()
            try:
                result=self.exchange('evaluate',self._serialize_input(encrypted,expected_tile),'result',tile['id'],while_pending=replay)
            finally:
                # The round trip contains replay; these two intervals overlap
                # and must never be added to estimate elapsed wall time.
                self.remote_seconds+=time.perf_counter()-start
            self.check_active()
            if not hmac.compare_digest(expected,result):
                raise ExecutionRejected('ciphertext-verification-failed')
            answer=ext.CompressedResultEncryptedMatrix.deserialize(result)
            self.admission.verified()
            self.observe('verified-tile',dict(sequence=self.sequence-1,projection=tile['name'],tileId=tile['id']))
            return answer
        except BaseException:
            self.admission.fail()
            self._release_projection_inputs()
            raise

    def decode(self,answer,tile,inputs,decrypt):
        """Every response reaching this bridge's caller was byte-verified by replay."""
        return decrypt(answer)

    def integrity_failed(self):
        """Replay rejects immediately; there is no deferred integrity outcome."""
        return False

    def require_integrity(self):
        """Replay rejects immediately; there is no deferred integrity outcome."""

    def finish(self):
        self.check_active()
        self.admission.complete()
        count = len(self.plan.schedule())*self.plan.definition()['steps']
        if self.sequence != count+1:
            raise ValueError('Incomplete full encrypted route')
        self.sent += send(self.child.stdin,self.plan.frame(self.session,self.key_id,self.sequence,'finish'),timeout=self.operation_timeout,cancelled=self.cancelled,max_payload_bytes=self.payload_limit)
        self.child.stdin.close()
        status=self.child.wait(timeout=180)
        self.check_active()
        if status != 0 or self.child.stdout.read(1):
            raise ValueError('Full evaluator failed to finish')

    def close(self):
        self.admission.fail()
        self._release_projection_inputs()
        try:
            if self.child.poll() is None:
                self.child.kill()
            self.child.wait()
        finally:
            self.key=None
            for stream in (self.child.stdin,self.child.stdout):
                if stream and not stream.closed:
                    stream.close()
