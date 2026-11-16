"""Complete real checkpoint metadata and explicit source scopes; no weight payloads or keys."""
import copy
import unittest
from unittest.mock import patch
from protocol import digest
from full_plan import canonical
from flash_tokenizer import SOURCE
from flash_source_scopes import BASE,TEXT,source_spec
from inventory_flash_text import assemble


class CompleteTextInventoryTests(unittest.TestCase):
    def metadata(self):
        names={name:value['shard'] for name,value in TEXT['tensors'].items()}
        metadata={'model.safetensors.index.json':dict(weight_map=names),'config.json':dict(text_config=dict(num_hidden_layers=48,num_experts=512,layer_types=['public-kind']*48))}
        headers={file:(copy.deepcopy(record),{}) for file,record in TEXT['shards'].items()}
        for name,value in TEXT['tensors'].items():headers[value['shard']][1][name]=dict(shape=value['shape'],dtype=value['dtype'],data_offsets=value['dataOffsets'])
        return metadata,headers

    def test_all_real_text_paths_have_bounded_geometry_and_complete_expert_banks(self):
        metadata,headers=self.metadata();self.assertEqual(assemble(metadata,headers),TEXT)
        self.assertEqual(TEXT['tensorCount'],1294);self.assertEqual(len(TEXT['shards']),131)
        self.assertEqual(TEXT['logicalTensorBytes'],353887799320);self.assertEqual(TEXT['wholeShardBytes'],360000192888)
        self.assertEqual(TEXT['tokenizerSourceId'],digest(canonical(SOURCE)))
        self.assertTrue(set(SOURCE['files']).issubset(TEXT['files']))
        self.assertEqual(TEXT['wholeSourceBytes'],TEXT['wholeShardBytes']+TEXT['auxiliarySourceBytes'])
        self.assertFalse(TEXT['weightsProvisioned']);self.assertFalse(TEXT['fullModelExecuted']);self.assertFalse(TEXT['securityQualified'])
        self.assertTrue(all(name.startswith('model.language_model.') or name=='lm_head.weight' for name in TEXT['tensors']))
        self.assertEqual(sum('.mlp.experts.' in name for name in TEXT['tensors']),96)

    def test_partial_out_of_bounds_or_different_materialization_is_not_an_inventory(self):
        for change in ['missing-shard','missing-expert','bad-offset','unsupported-dtype','different-model']:
            metadata,headers=self.metadata();name='model.language_model.layers.47.mlp.experts.gate_up_proj';file=TEXT['tensors'][name]['shard']
            if change=='missing-shard':headers.pop(file)
            elif change=='missing-expert':metadata['model.safetensors.index.json']['weight_map'].pop(name)
            elif change=='bad-offset':headers[file][1][name]['data_offsets']=[0,2**63]
            elif change=='unsupported-dtype':headers[file][1][name]['dtype']='advertised-opaque'
            else:metadata['config.json']['text_config']['num_hidden_layers']=1
            with self.subTest(change=change),self.assertRaises(ValueError):assemble(metadata,headers)

    def test_tokenizer_mismatch_or_missing_artifact_rejects_before_weight_materialization(self):
        metadata,headers=self.metadata()
        with patch('inventory_flash_text.TOKENIZER',dict(SOURCE,vocabSize=SOURCE['vocabSize']-1)):
            with self.assertRaises(ValueError):assemble(metadata,headers)
        from flash_weights import FlashSource
        visited=[]
        def verify(path,record):
            visited.append(path.name)
            if path.name=='tokenizer.json':raise ValueError('required tokenizer missing')
        with patch('flash_weights.verify_blob',side_effect=verify):
            with self.assertRaises(ValueError):FlashSource('/unused','text')
        self.assertIn('tokenizer.json',visited)
        self.assertFalse(any(name.endswith('.safetensors') for name in visited))

    def test_full_text_acquisition_is_explicit_and_legacy_component_scopes_do_not_expand(self):
        self.assertEqual(source_spec('moe'),BASE);self.assertNotIn('lm_head.weight',source_spec('moe')['tensors'])
        complete=source_spec('text');complete['tensors'].clear();self.assertEqual(source_spec('text'),TEXT)
        with self.assertRaises(ValueError):source_spec('advertised-full-model')
        from plan_backends import load_backend_plan
        for name in ['text','flash-text']:
            with self.assertRaises(ValueError):load_backend_plan('/unused',name)


if __name__=='__main__':unittest.main()
