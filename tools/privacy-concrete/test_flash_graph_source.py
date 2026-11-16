"""All-stage tiny BF16 graph assembly; no actual Flash checkpoint or crypto."""
import hashlib,json,os,tempfile,threading,unittest
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in Torch reference')
class GraphSourceTests(unittest.TestCase):
    def test_all_sources_and_local_tables_compose_complete_cached_graph(self):
        import torch
        from safetensors.torch import save_file
        from flash_weights import FlashSource
        from flash_graph_source import bind_graph
        from flash_streamed_moe import SourceLinear
        from flash_request_state import ClientContext
        from test_flash_decoder_source import DecoderSourceTests
        from test_flash_model_reference import FlashModelReferenceTests
        from test_flash_ple_local_reference import LocalPLETests
        from test_local_token_embedding_reference import fixture_rows
        with ExitStack() as stack:
            directory=lambda:Path(stack.enter_context(tempfile.TemporaryDirectory()))
            token_rows=fixture_rows(directory());ple_rows=LocalPLETests().fixture(directory())[6]
            stack.callback(token_rows.close);stack.callback(ple_rows.close)
            _,reference,parts,_=FlashModelReferenceTests().fixture(ple_rows,token_rows)
            decoders=[];specs=[]
            for index in range(3):
                _,_,layer,source,spec=DecoderSourceTests().fixture(directory(),index,ple_rows)
                for name in list(spec['tensors']):
                    if '.ngram_embedding.' in name:del spec['tensors'][name]
                parts[1][index]=layer;decoders.append(source);specs.append(spec)
            tensors={}
            for prefix,module in [('model.language_model.hyper_connection_mixer.',parts[2]),('lm_head.',parts[3])]:
                for name,value in module.state_dict().items():tensors[prefix+name]=value.detach().to(torch.bfloat16)
                module.load_state_dict({name:value.detach().to(torch.bfloat16).float() for name,value in module.state_dict().items()})
            root=directory();shard=root/'public.safetensors';save_file(tensors,str(shard))
            (root/'config.json').write_text(json.dumps({'text_config':vars(reference.config)}))
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{name:shard.name for name in tensors}}))
            def record(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            output_spec=dict(files={name:record(root/name) for name in ['config.json','model.safetensors.index.json']},shards={shard.name:record(shard)},
                tensors={name:dict(shard=shard.name,shape=list(value.shape),dtype='BF16') for name,value in tensors.items()})
            with patch('flash_weights.source_spec',return_value=output_spec):output=FlashSource(root)
            def stage(part,index=None):return output_spec if part=='output' else specs[index]
            with patch('flash_graph_source.TEXT',{'layerCount':3}),patch('flash_graph_source.installed_inventory',return_value=token_rows.spec), \
                 patch('flash_graph_source.TABLE',ple_rows.spec),patch('flash_decoder_source.TABLE',ple_rows.spec), \
                 patch('flash_graph_source.staging_spec',side_effect=stage),patch('flash_decoder_source.staging_spec',side_effect=stage), \
                 patch('flash_streamed_moe.staging_spec',side_effect=stage),patch('flash_decoder_source.resolve_text_config',return_value=reference.config):
                graph=bind_graph(decoders,output,token_rows,ple_rows,reference.reference_path,
                    context_limit=12,max_lookup_rows=12,max_ple_lookup_bytes=65536)
            self.assertTrue(all(module.weight.is_meta for module in graph.modules() if isinstance(module,SourceLinear)))
            context=ClientContext('tenant','privacy',graph.identity,'key','request');state=graph.new_state(context)
            try:
                prefix=torch.tensor([[1,2]])
                for value in [prefix,torch.tensor([[3]])]:
                    if value.shape[1]==1:prefix=torch.cat([prefix,value],dim=1)
                    actual=graph(value,state,context,execution='public-reference')
                    expected=FlashModelReferenceTests().full_prefix(torch,graph,parts,prefix)
                    torch.testing.assert_close(actual,expected[:,-value.shape[1]:],rtol=3e-4,atol=3e-5)
                output.closed=True
                with self.assertRaises(ValueError):graph(torch.tensor([[4]]),state,context,execution='public-reference')
                self.assertTrue(state.failed)
            finally:state.close()
            self.assertFalse(token_rows.closed);self.assertFalse(ple_rows.closed)

    def test_precancelled_factory_does_not_touch_sources(self):
        from flash_graph_source import bind_graph
        event=threading.Event();event.set()
        with self.assertRaises(InterruptedError):bind_graph(None,None,None,None,'unused',context_limit=1,max_lookup_rows=1,max_ple_lookup_bytes=1,cancelled=event)

if __name__=='__main__':unittest.main()
