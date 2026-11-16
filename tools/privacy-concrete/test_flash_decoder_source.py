"""Whole decoder binding with real small BF16 files, not full Flash qualification."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in pinned source/Torch reference')
class DecoderSourceTests(unittest.TestCase):
    def fixture(self,directory,index,storage=None):
        import torch
        from safetensors.torch import save_file
        from test_flash_model_reference import FlashModelReferenceTests
        from flash_weights import FlashSource
        torch,graph,reference,_=FlashModelReferenceTests().fixture(storage)
        layer=reference[1][index];prefix=f'model.language_model.layers.{index}.'
        tensors={prefix+name:value.detach().to(torch.bfloat16) if value.dtype==torch.float32 else value for name,value in layer.state_dict().items()}
        # Compare against the exact BF16 source representation, not original FP32.
        layer.load_state_dict({name:tensors[prefix+name].to(value.dtype) for name,value in layer.state_dict().items()})
        root=Path(directory);shard=root/'public.safetensors';save_file(tensors,str(shard))
        (root/'config.json').write_text(json.dumps({'text_config':vars(graph.config)}))
        (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{name:shard.name for name in tensors}}))
        def record(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        spec=dict(files={name:record(root/name) for name in ['config.json','model.safetensors.index.json']},shards={shard.name:record(shard)},
            tensors={name:dict(shard=shard.name,shape=list(value.shape),dtype='BF16' if value.dtype==torch.bfloat16 else 'I64') for name,value in tensors.items()})
        with patch('flash_weights.source_spec',return_value=spec):source=FlashSource(root)
        return torch,graph,layer,source,spec

    def test_delta_decoder_streams_all_linear_weights_and_matches_materialized_source(self):
        from flash_decoder_source import bind_decoder
        from flash_streamed_moe import SourceLinear
        from flash_rotary import text_rotary
        with tempfile.TemporaryDirectory() as directory:
            torch,graph,reference,source,spec=self.fixture(directory,0)
            with patch('flash_decoder_source.staging_spec',return_value=spec),patch('flash_streamed_moe.staging_spec',return_value=spec), \
                 patch('flash_decoder_source.resolve_text_config',return_value=graph.config):
                layer,config=bind_decoder(source,graph.reference_path,0)
            streamed=[module for module in layer.modules() if isinstance(module,SourceLinear)]
            self.assertGreater(len(streamed),10);self.assertTrue(all(module.weight.device.type=='meta' for module in streamed))
            value=torch.linspace(-.3,.2,16).reshape(1,2,8)
            kwargs=dict(position_embeddings=text_rotary(graph.reference_path,config,2),attention_mask=None,conv_mask=None,cache=None,ple_input_ids=torch.tensor([[1,2]]))
            with torch.no_grad():torch.testing.assert_close(layer(value,**kwargs),reference(value,**kwargs),rtol=2e-5,atol=2e-6)
            source.closed=True
            with self.assertRaises(ValueError):layer(value,**kwargs)

    def test_ple_indexed_decoder_integrates_with_request_owned_local_rows(self):
        from flash_decoder_source import bind_decoder
        from flash_model_graph import ClientFlashGraph
        from flash_request_state import ClientContext
        from test_flash_ple_local_reference import LocalPLETests
        from test_flash_model_reference import FlashModelReferenceTests
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as row_directory:
            storage=LocalPLETests().fixture(row_directory)[6]
            try:
                torch,graph,reference,source,spec=self.fixture(directory,1,storage)
                removed=[name for name in spec['tensors'] if '.ngram_embedding.' in name]
                for name in removed:del spec['tensors'][name]
                with patch('flash_decoder_source.staging_spec',return_value=spec),patch('flash_streamed_moe.staging_spec',return_value=spec), \
                     patch('flash_decoder_source.resolve_text_config',return_value=graph.config),patch('flash_decoder_source.TABLE',storage.spec):
                    layer,config=bind_decoder(source,graph.reference_path,1,local_rows=storage,max_lookup_bytes=65536)
                # Use fresh materialized sibling layers and actual request-local cache composition.
                _,_,parts,_=FlashModelReferenceTests().fixture(storage)
                parts[1][1]=reference
                expected_parts=__import__('copy').deepcopy(parts)
                parts[1][1]=layer
                complete=ClientFlashGraph(graph.reference_path,config,*parts,context_limit=12)
                context=ClientContext('tenant','privacy',complete.identity,'key','request');state=complete.new_state(context)
                tokens=torch.tensor([[1,2]],dtype=torch.int64)
                actual=complete(tokens,state,context,execution='public-reference')
                expected=FlashModelReferenceTests().full_prefix(torch,complete,expected_parts,tokens)
                torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-5)
                self.assertIs(state.ple[1].local_rows_lease.storage,storage)
                state.close();self.assertFalse(storage.closed)
            finally:storage.close()

    def test_cancelled_binding_reads_no_source_or_reference(self):
        import threading
        from flash_decoder_source import bind_decoder
        event=threading.Event();event.set()
        with self.assertRaises(InterruptedError):bind_decoder(None,'unused',0,cancelled=event)

    def test_mutated_configuration_bytes_are_rejected(self):
        from flash_decoder_source import bind_decoder
        with tempfile.TemporaryDirectory() as directory:
            _,graph,_,source,spec=self.fixture(directory,0)
            (Path(directory)/'config.json').write_text('{}')
            with patch('flash_decoder_source.staging_spec',return_value=spec):
                with self.assertRaisesRegex(ValueError,'Pinned decoder configuration'):bind_decoder(source,graph.reference_path,0)

    def test_unaccounted_stage_tensor_is_rejected(self):
        from flash_decoder_source import bind_decoder
        with tempfile.TemporaryDirectory() as directory:
            _,graph,_,source,spec=self.fixture(directory,0)
            spec['tensors']['model.language_model.layers.0.mlp.unused.weight']=dict(spec['tensors']['model.language_model.layers.0.mlp.gate.weight'])
            with patch('flash_decoder_source.staging_spec',return_value=spec),patch('flash_streamed_moe.staging_spec',return_value=spec), \
                 patch('flash_decoder_source.resolve_text_config',return_value=graph.config):
                with self.assertRaisesRegex(ValueError,'Incomplete decoder'):bind_decoder(source,graph.reference_path,0)

if __name__=='__main__':unittest.main()
