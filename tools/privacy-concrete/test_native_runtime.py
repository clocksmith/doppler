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
        self.assertEqual(accelerated['dependencies']['concrete-ml-extensions'],'0.1.9+ntt.7')
        legacy['steps']=1;parameters['parameters']['bits_reserved_for_computation']=1
        again,_,_,fresh=qwen_backend('qwen-all-linear')
        self.assertEqual(again['steps'],4);self.assertEqual(fresh['parameters']['bits_reserved_for_computation'],27)
        self.assertEqual(len(ntt_source_id()),64)
        self.assertEqual(ntt_parameters()['nativeSourceId'],ntt_source_id())
        raw=canonical(dict(modelId='a'*64))
        self.assertNotEqual(FullPlan(raw).frame('s','k',1,'evaluate'),FullNttPlan(raw).frame('s','k',1,'evaluate'))

    def test_component_variant_is_explicit_and_does_not_change_stock_parameters(self):
        from flash_component_plan import component_parameters
        stock=component_parameters({})
        fast=component_parameters({'nativeVariant':'cpu-exact-ntt/v1'})
        self.assertEqual(stock['evaluationBackend'],'cpu');self.assertEqual(fast['evaluationBackend'],'cpu-exact-ntt')
        self.assertEqual(stock['parameters'],fast['parameters'])
        with self.assertRaises(ValueError):component_parameters({'nativeVariant':'advertised-unknown'})

    def test_high_precision_is_an_explicit_protocol_and_frame_budget_without_changing_defaults(self):
        high,_,_,parameters=qwen_backend('qwen-all-linear-ntt-48')
        low,_,_,unchanged=qwen_backend('qwen-all-linear-ntt')
        self.assertNotEqual(high['protocol'],low['protocol'])
        self.assertEqual(parameters['parameters']['output_storage_ciphertext_modulus'],48)
        self.assertEqual(parameters['parameters']['packing_ks_level'],4)
        self.assertEqual(parameters['maxPayloadBytes'],300000000)
        self.assertEqual(unchanged['parameters']['output_storage_ciphertext_modulus'],26)
        self.assertNotIn('maxPayloadBytes',unchanged)
        self.assertEqual(parameters['parameters']['glwe_encryption_noise_distribution_stdev'],unchanged['parameters']['glwe_encryption_noise_distribution_stdev'])

    def test_unknown_full_plan_fields_reject_before_library_or_file_materialization(self):
        import json
        import tempfile
        from pathlib import Path
        from full_plan import load_plan
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'plan.json').write_text(json.dumps({'advertisedAttestation':True},sort_keys=True,separators=(',',':')))
            with patch('full_plan.implementation_versions',side_effect=AssertionError('unknown plan entered implementation admission')):
                with self.assertRaises(ValueError):load_plan(directory)

    def test_twelve_bit_profile_uses_a_distinct_installed_message_domain(self):
        policy,_,_,parameters=qwen_backend('qwen-all-linear-ntt12')
        self.assertEqual(policy['quantizationBits'],12);self.assertEqual(parameters['quantizationBits'],12)
        self.assertEqual(parameters['weightEncoding'],'signed-int12-in-int16')
        self.assertEqual(parameters['parameters']['bits_reserved_for_computation'],35)
        self.assertEqual(parameters['maxPayloadBytes'],300000000)

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
