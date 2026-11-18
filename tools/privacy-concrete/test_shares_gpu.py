"""GPU evaluator for the shares route against the CPU reference. Needs Node and a WebGPU binding:
set DOPPLER_SHARES_WEBGPU_MODULE to the binding's index.js. Skipped otherwise."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np=None

from full_status import ExecutionRejected
import shares

MODULE=os.environ.get('DOPPLER_SHARES_WEBGPU_MODULE')
NODE=shutil.which('node')
SEED=bytes(range(32))
P=shares.PRIME
SHAPES={'small':(48,20),'odd':(47,21),'wide':(896,301),'deep':(5000,70)}


@unittest.skipIf(np is None or not MODULE or not NODE,'NumPy, Node and DOPPLER_SHARES_WEBGPU_MODULE are required')
class GpuSharesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory()
        generator=np.random.default_rng(5)
        cls.weights={}
        rows=[]
        for name,(inputs,outputs) in SHAPES.items():
            weights=generator.integers(-2048,2048,size=(inputs,outputs),dtype=np.int16)
            weights[0,0],weights[1,0],weights[-1,-1]=-32768,32767,-32768
            if name=='deep':weights[:,3]=32767;weights[:,4]=-32768
            data=weights.astype('<i2').tobytes();identity=hashlib.sha256(data).hexdigest()
            (Path(cls.directory.name)/(identity+'.i16')).write_bytes(data)
            cls.weights[name]=weights
            rows.append(dict(name=name,inputWidth=inputs,outputWidth=outputs,weightId=identity,file=identity+'.i16',dtype='<i2'))
        (Path(cls.directory.name)/'shares-manifest.json').write_text(json.dumps(dict(schema='doppler.two-server-shares-manifest/v1',projections=rows)))
        here=Path(shares.__file__).parent
        cls.gpu=[NODE,str(here/'shares_evaluator_gpu.js'),'--cache-dir',cls.directory.name,'--webgpu-module',MODULE]
        cls.cpu=[sys.executable,str(here/'shares_evaluator.py'),'--cache-dir',cls.directory.name]

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_gpu_product_equals_the_cpu_reference_for_field_elements_at_every_extreme(self):
        generator=np.random.default_rng(9)
        evaluator=shares.ProcessEvaluator(self.gpu,'gpu',timeout=120)
        try:
            for name,(inputs,outputs) in SHAPES.items():
                for count in (1,5):
                    rows=shares._field(generator.bytes(8*count*inputs),(count,inputs)).copy()
                    rows[0,:]=P-1;rows[-1,0]=0
                    if count>1:rows[1,:]=np.uint64(P);rows[2,:]=np.uint64((1<<64)-1);rows[3,:]=0
                    evaluator.send(name,rows)
                    got=evaluator.receive(count,outputs)
                    np.testing.assert_array_equal(got,shares.field_product(rows,self.weights[name]),(name,count))
                    self.assertTrue(bool((got<np.uint64(P)).all()))
        finally:
            evaluator.close()

    def route(self,commands):
        anchors={name:shares.build_anchor(SEED,name,weights) for name,weights in self.weights.items()}
        geometry={name:shape for name,shape in SHAPES.items()}
        return shares.ShareRoute(tuple(shares.ProcessEvaluator(command,label,timeout=120) for command,label in zip(commands,'ab')),SEED,anchors,geometry=geometry)

    def test_a_complete_route_on_two_gpu_evaluators_and_on_one_gpu_with_one_cpu_is_exact(self):
        generator=np.random.default_rng(13)
        for commands in ((self.gpu,self.gpu),(self.gpu,self.cpu)):
            route=self.route(commands)
            try:
                for name,(inputs,outputs) in SHAPES.items():
                    for count in (8,1):
                        inputs_rows=generator.integers(0,4096,size=(count,inputs),dtype=np.uint64)
                        inputs_rows[0,:]=4095
                        expected=inputs_rows.astype(np.int64)@self.weights[name].astype(np.int64)
                        bits=12 if name!='deep' else 12
                        got=route.linear(name,inputs_rows,outputs,bits) if shares.product_bound(inputs,bits)>np.abs(expected).max() else None
                        if got is not None:np.testing.assert_array_equal(got,expected,(name,count))
                route.finish()
            finally:
                route.close()
            route.require_integrity()
            self.assertGreater(route.calls,0)

    def test_gpu_evaluator_refuses_a_changed_coefficient_file_and_an_unknown_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            for item in Path(self.directory.name).iterdir():shutil.copy(item,directory)
            manifest=json.loads((Path(directory)/'shares-manifest.json').read_text())
            (Path(directory)/manifest['projections'][0]['file']).write_bytes(bytes(48*20*2))
            command=[*self.gpu[:3],directory,*self.gpu[4:]]
            route=shares.ShareRoute((shares.ProcessEvaluator(command,'a',timeout=60),shares.ProcessEvaluator(command,'b',timeout=60)),SEED,
                                    {'small':shares.build_anchor(SEED,'small',self.weights['small'])})
            try:
                with self.assertRaises(ExecutionRejected) as caught:route.linear('small',np.zeros((1,48),dtype=np.uint64),20,12)
                self.assertEqual(caught.exception.code,'execution-failed')
            finally:
                route.close()
        evaluator=shares.ProcessEvaluator(self.gpu,'gpu',timeout=60)
        try:
            evaluator.send('missing',np.zeros((1,48),dtype=np.uint64))
            with self.assertRaises(ConnectionError):evaluator.receive(1,20)
        finally:
            evaluator.close()

    def test_gpu_workspace_reuses_buffers_across_growing_shrinking_and_multi_chunk_calls(self):
        generator=np.random.default_rng(21)
        with tempfile.TemporaryDirectory() as directory:
            weights_wide=generator.integers(-32768,32768,size=(5000,1400),dtype=np.int16)
            weights_small=generator.integers(-128,128,size=(31,13),dtype=np.int8)
            weights_small[0,0],weights_small[-1,-1]=-128,127
            wide_bytes=weights_wide.astype('<i2').tobytes();wide_id=hashlib.sha256(wide_bytes).hexdigest()
            small_bytes=weights_small.tobytes();small_id=hashlib.sha256(small_bytes).hexdigest()
            (Path(directory)/(wide_id+'.i16')).write_bytes(wide_bytes)
            (Path(directory)/(small_id+'.i8')).write_bytes(small_bytes)
            manifest=dict(schema='doppler.two-server-shares-manifest/v1',projections=[
                dict(name='chunked',inputWidth=5000,outputWidth=1400,weightId=wide_id,file=wide_id+'.i16',dtype='<i2'),
                dict(name='tiny_i8',inputWidth=31,outputWidth=13,weightId=small_id,file=small_id+'.i8',dtype='|i1'),
            ])
            (Path(directory)/'shares-manifest.json').write_text(json.dumps(manifest))
            command=[*self.gpu[:3],directory,*self.gpu[4:]]
            evaluator=shares.ProcessEvaluator(command,'gpu',timeout=120)
            try:
                # 120 * 5000 = 600,000 inner ops/col -> 640 cols/chunk -> 3 column chunks; followed by tiny 1-row i8 call.
                schedule=[('tiny_i8',weights_small,1),('chunked',weights_wide,120),('tiny_i8',weights_small,1),
                          ('chunked',weights_wide,3),('tiny_i8',weights_small,7)]
                for name,matrix,count in schedule:
                    inputs= matrix.shape[0]
                    rows=shares._field(generator.bytes(8*count*inputs),(count,inputs)).copy()
                    rows[0,:]=P-1
                    evaluator.send(name,rows)
                    got=evaluator.receive(count,matrix.shape[1])
                    np.testing.assert_array_equal(got,shares.field_product(rows,matrix),(name,count))
            finally:
                evaluator.close()

    def test_gpu_evaluator_rejects_symlinks_invalid_geometry_and_exits_cleanly_on_truncated_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            data=self.weights['small'].astype('<i2').tobytes()
            identity=hashlib.sha256(data).hexdigest()
            real=Path(directory)/(identity+'.i16');real.write_bytes(data)
            (Path(directory)/'link.i16').symlink_to(real)
            manifest=dict(schema='doppler.two-server-shares-manifest/v1',projections=[
                dict(name='linked',inputWidth=48,outputWidth=20,weightId=identity,file='link.i16',dtype='<i2'),
                dict(name='too_wide',inputWidth=1<<21,outputWidth=4,weightId=identity,file=identity+'.i16',dtype='<i2'),
                dict(name='zero_out',inputWidth=48,outputWidth=0,weightId=identity,file=identity+'.i16',dtype='<i2'),
            ])
            (Path(directory)/'shares-manifest.json').write_text(json.dumps(manifest))
            command=[*self.gpu[:3],directory,*self.gpu[4:]]
            for bad_name,width in (('linked',48),('too_wide',1<<21),('zero_out',48)):
                evaluator=shares.ProcessEvaluator(command,'gpu',timeout=60)
                try:
                    # Send header + minimal or zero payload; invalid geometry must fail before accepting work.
                    encoded=bad_name.encode()
                    evaluator.sequence+=1
                    head=shares.REQUEST_HEAD.pack(shares.REQUEST,evaluator.sequence,len(encoded),1,width)+encoded
                    evaluator.process.stdin.write(head)
                    if bad_name=='linked':evaluator.process.stdin.write(bytes(8*48))
                    evaluator.process.stdin.flush()
                    with self.assertRaises(ConnectionError):evaluator.receive(1,20)
                finally:
                    evaluator.close()

        # Truncated stdin frames at every boundary must exit 0 without an unhandled rejection.
        full_name=b'small'
        full_head=shares.REQUEST_HEAD.pack(shares.REQUEST,1,len(full_name),2,48)
        for partial in (b'SH',full_head[:10],full_head+b'sm',full_head+full_name+bytes(16)):
            completed=subprocess.run(self.gpu,input=partial,capture_output=True,timeout=60,check=False)
            stderr=completed.stderr.decode(errors='replace')
            self.assertEqual(completed.returncode,0,stderr)
            self.assertNotIn('end of stream',stderr)
            self.assertNotIn('shares evaluator:',stderr)


if __name__=='__main__':
    unittest.main()
