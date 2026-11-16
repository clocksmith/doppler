"""No variant downgrade or global policy mutation from research backend selection."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from plan_backends import qwen_backend
from full_plan import FullPlan,canonical
from full_ntt_plan import FullNttPlan
from native_runtime import matrix_evaluator,ntt_parameters,ntt_source_id


class NativeSelectionTests(unittest.TestCase):
    def test_variant_changes_protocol_and_versions_without_mutating_legacy_policy(self):
        legacy,_,_,parameters=qwen_backend('qwen-all-linear')
        fast,_,_,accelerated=qwen_backend('qwen-all-linear-ntt')
        self.assertNotEqual(legacy['protocol'],fast['protocol'])
        self.assertEqual(parameters['dependencies']['concrete-ml-extensions'],'0.1.9')
        self.assertEqual(accelerated['dependencies']['concrete-ml-extensions'],'0.1.9+ntt.1')
        legacy['steps']=1;parameters['parameters']['bits_reserved_for_computation']=1
        again,_,_,fresh=qwen_backend('qwen-all-linear')
        self.assertEqual(again['steps'],4);self.assertEqual(fresh['parameters']['bits_reserved_for_computation'],27)
        self.assertEqual(len(ntt_source_id()),64)
        self.assertEqual(ntt_parameters()['nativeSourceId'],ntt_source_id())
        raw=canonical(dict(modelId='a'*64))
        self.assertNotEqual(FullPlan(raw).frame('s','k',1,'evaluate'),FullNttPlan(raw).frame('s','k',1,'evaluate'))

    def test_unknown_evaluator_does_not_import_or_invoke_a_native_module(self):
        plan=SimpleNamespace(definition=lambda:dict(parameters=dict(evaluationBackend='advertised-unknown')))
        with patch('native_runtime.importlib.import_module',side_effect=AssertionError('native module imported')):
            with self.assertRaises(ValueError):matrix_evaluator(plan)
        with self.assertRaises(ValueError):qwen_backend('flash-full-unsupported')

    def test_stock_native_module_remains_the_explicit_legacy_evaluator(self):
        expected=object();native=SimpleNamespace(cpu_matrix_multiplication=expected)
        plan=SimpleNamespace(definition=lambda:dict(parameters=dict(evaluationBackend='cpu')))
        with patch('native_runtime.importlib.import_module',return_value=native):self.assertIs(matrix_evaluator(plan),expected)


if __name__=='__main__':unittest.main()
