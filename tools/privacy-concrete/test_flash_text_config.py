"""Pinned real configuration/complete geometry; no model weights or inference."""
import copy
import json
import os
from pathlib import Path
import unittest
from flash_text_config import resolve_text_config


class FlashConfigBoundaryTests(unittest.TestCase):
    def test_unpinned_configuration_cannot_execute_any_declarations(self):
        with self.assertRaises(ValueError):resolve_text_config({},b"raise RuntimeError('untrusted source executed')",attention_implementation='eager')

    def test_unsupported_dispatch_or_method_override_fails_before_source_execution(self):
        for raw,implementation in [({},'flash_attention_2'),({'__post_init__':'untrusted'},'eager'),({'validate_architecture':None},'eager'),({'_attn_implementation':'unknown'},'eager')]:
            with self.assertRaises(ValueError):resolve_text_config(raw,b'not executed',attention_implementation=implementation)


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in pinned source/meta-Torch assembly')
class FlashCheckpointAssemblyTests(unittest.TestCase):
    def fixture(self):
        root=Path(os.environ['DOPPLER_FLASH_SOURCE_DIR']);raw=json.loads((root/'checkpoint-config.json').read_text())['text_config']
        return root,raw,(root/'configuration_qwen4_exp.py').read_bytes()

    def test_actual_checkpoint_resolves_source_aliases_defaults_without_mutating_raw_configuration(self):
        root,raw,source=self.fixture();before=copy.deepcopy(raw)
        config=resolve_text_config(raw,source,attention_implementation='eager')
        self.assertEqual(raw,before);self.assertNotIn('full_attention',config.layer_types)
        self.assertEqual(config.layer_types.count('indexed_attention'),raw['layer_types'].count('full_attention'))
        self.assertEqual(config.seed,1234);self.assertEqual(config.number_of_conv_states,3)
        self.assertEqual(len(config.layer_types),config.num_hidden_layers)
        invalid=copy.deepcopy(raw);invalid['layer_types'][0]='unsupported'
        with self.assertRaises(ValueError):resolve_text_config(invalid,source,attention_implementation='eager')
        for layers in [None,[],raw['layer_types'][:-1]]:
            invalid=copy.deepcopy(raw);invalid['layer_types']=layers
            with self.assertRaises(ValueError):resolve_text_config(invalid,source,attention_implementation='eager')
        invalid=copy.deepcopy(raw);invalid['ple_layer_ids']=[4]
        with self.assertRaises(ValueError):resolve_text_config(invalid,source,attention_implementation='eager')

    def test_all_actual_checkpoint_tensor_destinations_match_complete_48_layer_meta_assembly(self):
        from flash_model_assembly import assemble_geometry,TEXT
        root,raw,source=self.fixture();model,config,report=assemble_geometry(root/'checkpoint-config.json',root/'modeling_qwen4_exp.py')
        self.assertEqual(report['layerCount'],TEXT['layerCount'])
        self.assertEqual(report['checkpointTensors'],TEXT['tensorCount']);self.assertTrue(report['allTensorGeometryMatched'])
        self.assertEqual(report['packedExpertBanks'],2*config.num_hidden_layers)
        self.assertFalse(report['weightsMaterialized']);self.assertFalse(report['forwardExecuted']);self.assertFalse(report['securityQualified'])
        self.assertTrue(all(value.device.type=='meta' for value in model.state_dict().values()))


if __name__=='__main__':unittest.main()
