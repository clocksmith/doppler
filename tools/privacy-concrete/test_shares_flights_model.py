"""The projection module's grouped-flight path, exercised without the homomorphic toolchain or model weights.

full_model.py imports a quantization executor from an optional package. This test stands a small exact stand-in
in its place so the module's own control flow can run anywhere: who is quantized, how often, what is sent in one
flight, and that a grouped projection returns exactly what an ungrouped one does.
"""
import importlib
import sys
import types
import unittest

try:
    import numpy as np
    import torch
    from torch import nn
except ImportError:
    np=torch=nn=None


def load_full_model():
    names=['concrete','concrete.ml','concrete.ml.quantization','concrete.ml.quantization.linear_op_glwe_backend','concrete_ml_extensions']
    added=[name for name in names if name not in sys.modules]
    for name in added:sys.modules[name]=types.ModuleType(name)
    sys.modules['concrete.ml.quantization.linear_op_glwe_backend'].GLWELinearLayerExecutor=object
    try:
        return importlib.import_module('full_model')
    except Exception as error:
        raise unittest.SkipTest(f'full_model needs more than the stand-in provides: {error}')
    finally:
        for name in added:sys.modules.pop(name,None)


class Executor:
    """Exact stand-in: unsigned integer inputs pass through, the product is returned unchanged."""
    def __init__(self):self.quantized=[]
    def _dynamic_input_quantization(self,value,context,_):
        self.quantized.append(context.label)
        return value.reshape(-1,value.shape[-1]).clone(),1.0,0,value.shape
    def _apply_correction_and_dequantize(self,raw,q_input,zero,zero_point,sum_weights,width,scale,weight_scale):
        return raw.to(torch.float64)


class Route:
    def __init__(self,weights):self.weights,self.singles,self.flights=weights,[],[]
    def linear(self,name,inputs,width,bits):
        self.singles.append(name);return inputs.astype(np.int64)@self.weights[name]
    def linear_many(self,calls):
        self.flights.append([call[0] for call in calls]);return [call[1].astype(np.int64)@self.weights[call[0]] for call in calls]


@unittest.skipIf(torch is None,'PyTorch is not installed in the dependency-free smoke job')
class GroupedFlightProjectionTest(unittest.TestCase):
    def setUp(self):
        self.module=load_full_model()
        generator=np.random.default_rng(3)
        self.names=['b.self_attn.q_proj','b.self_attn.k_proj','b.self_attn.v_proj','b.self_attn.o_proj']
        self.weights={name:generator.integers(-9,9,size=(6,4 if 'k_proj' in name else 5),dtype=np.int64) for name in self.names}
        self.executor=Executor()
        self.route=Route(self.weights)
        self.value=torch.tensor(generator.integers(0,4096,size=(1,3,6)),dtype=torch.float64)

    def projection(self,name):
        module=self.module.FullLinear.__new__(self.module.FullLinear);nn.Module.__init__(module)
        module.name,module.mode,module.executor,module.shares=name,'shared',self.executor,self.route
        module.context=types.SimpleNamespace(label=name)
        module.definition=dict(inputWidth=6,outputWidth=self.weights[name].shape[1])
        module.encoding=dict(quantizationBits=12)
        module.remote,module.prepared_bias=None,None
        module.scale=module.zero_point=module.sum_weights=None
        return module

    def run_block(self,grouped):
        q,k,v,o=(self.projection(name) for name in self.names)
        if grouped:q.flight_partners=[k,v]
        return [module(self.value) for module in (q,k,v,o)],(q,k,v,o)

    def test_grouped_projections_return_what_separate_ones_do_with_one_flight_and_one_quantization_each(self):
        separate,_=self.run_block(False)
        self.assertEqual((self.route.singles,self.route.flights),(self.names,[]))
        self.assertEqual(self.executor.quantized,self.names)
        self.route.singles.clear();self.executor.quantized.clear()
        grouped,modules=self.run_block(True)
        for got,want in zip(grouped,separate):
            self.assertEqual(got.shape,want.shape);self.assertTrue(torch.equal(got,want))
        # Query, key and value went out together; the output projection, which needs their result, went alone.
        self.assertEqual(self.route.flights,[self.names[:3]])
        self.assertEqual(self.route.singles,[self.names[3]])
        self.assertEqual(sorted(self.executor.quantized),sorted(self.names))
        self.assertTrue(all(module.flight_answer is None for module in modules))

    def test_an_answer_held_for_a_different_tensor_is_not_used(self):
        q,k,v,_=(self.projection(name) for name in self.names)
        q.flight_partners=[k,v]
        q(self.value)
        other=self.value+1
        got=k(other)
        want=torch.from_numpy(other.reshape(-1,6).numpy().astype(np.int64)@self.weights[k.name]).to(torch.float64).view(1,3,4)
        self.assertTrue(torch.equal(got,want))
        self.assertEqual(self.route.singles,[k.name])
        self.assertIsNone(k.flight_answer)

    def test_a_partner_outside_the_route_or_with_another_input_width_is_left_out(self):
        q,k,v,_=(self.projection(name) for name in self.names)
        k.shares=Route(self.weights);v.definition=dict(v.definition,inputWidth=7)
        q.flight_partners=[k,v]
        q(self.value)
        self.assertEqual((self.route.singles,self.route.flights),([q.name],[]))


if __name__=='__main__':unittest.main()
