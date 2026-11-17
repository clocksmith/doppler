"""GPU evaluator for the shares route against the CPU reference. Needs Node and a WebGPU binding:
set DOPPLER_SHARES_WEBGPU_MODULE to the binding's index.js. Skipped otherwise."""
import hashlib
import json
import os
from pathlib import Path
import shutil
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


if __name__=='__main__':
    unittest.main()
