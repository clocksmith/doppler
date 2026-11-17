"""Actual BF16 source appraisal, reconstruction, cache lifetime and failure cleanup.

Small synthetic data; no full checkpoint or cryptographic qualification.
"""
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch,Mock


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in Torch checkpoint source')
class ProviderCheckpointTests(unittest.TestCase):
    def fixture(self,stack):
        from test_flash_stream_storage import FlashStreamStorageTests
        from flash_provider_source import ProviderCheckpointSource
        helper=FlashStreamStorageTests();_,graph,prepared,root,plan,binding=helper.fixture(stack,rebuild=True)
        scratch=Path(stack.enter_context(tempfile.TemporaryDirectory()))
        source=ProviderCheckpointSource(helper.provider_source_directory,scratch,plan);stack.callback(source.close)
        return helper,prepared,root,plan,binding,scratch,source

    def test_owned_projection_reused_then_evicted_and_same_size_corruption_rejects(self):
        from public_weights import PublicTileReader
        with ExitStack() as stack:
            _,_,_,plan,_,scratch,source=self.fixture(stack)
            sibling=scratch/'keep';sibling.write_text('keep');first,second=plan.schedule()[:2]
            with PublicTileReader(scratch,plan,open_projection=source.open_projection,check_active=source.require_tile_source) as reader:
                original=reader.read(first);self.assertEqual(reader.read(first),original);self.assertEqual(source.materializations,1)
                owned=list(scratch.glob('provider-public-projection-*'));self.assertEqual(len(owned),1)
                reader.read(second);self.assertEqual(source.materializations,2);self.assertFalse(owned[0].exists())
                current=next(scratch.glob('provider-public-projection-*'));path=next(current.glob('*.i16'))
                raw=path.read_bytes();path.write_bytes(bytes([raw[0]^1])+raw[1:])
                # Reader returns owned public bytes; evaluator's independent
                # before-parser digest check must reject this mutation.
                from protocol import digest
                self.assertNotEqual(digest(reader.read(second)),plan.definition()['publicTileDigests'][second['id']])
            self.assertEqual(list(scratch.iterdir()),[sibling]);self.assertEqual(source.live_cache_bytes,0)

    def test_projection_budget_is_reserved_during_preparation_not_only_after_publication(self):
        from public_weights import PublicTileReader
        with ExitStack() as stack:
            _,_,_,plan,_,scratch,source=self.fixture(stack);tile=plan.schedule()[0]
            row=plan.definition()['projections'][0];quantize=source.executor._per_channel_weight_quantization
            def reenter(*args):
                with self.assertRaisesRegex(ValueError,'already active'):
                    with source.open_projection(row,source.encoding):self.fail('Second cache preparation escaped its budget')
                self.assertEqual(len(list(scratch.glob('provider-public-projection-*'))),1)
                return quantize(*args)
            with patch.object(source.executor,'_per_channel_weight_quantization',side_effect=reenter):
                with PublicTileReader(scratch,plan,open_projection=source.open_projection,check_active=source.require_tile_source) as reader:reader.read(tile)
            self.assertFalse(list(scratch.iterdir()));self.assertEqual(source.live_cache_bytes,0)

    def test_cancel_bad_source_or_quantizer_mismatch_cannot_leave_owned_cache(self):
        from public_weights import PublicTileReader
        for failure in ('cancel','source','quantizer','space'):
            with self.subTest(failure=failure),ExitStack() as stack:
                helper,_,_,plan,_,scratch,source=self.fixture(stack)
                tile=plan.schedule()[0];sibling=scratch/'keep';sibling.write_text('keep')
                if failure=='cancel':source.cancelled=threading.Event();source.cancelled.set()
                elif failure=='source':
                    path=helper.provider_source_directory/'public.safetensors';raw=path.read_bytes();path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
                elif failure=='quantizer':
                    quantize=source.executor._per_channel_weight_quantization
                    def wrong(*args):
                        values,*metadata=quantize(*args);values=values.clone();values[0,0]+=1;return (values,*metadata)
                    stack.enter_context(patch.object(source.executor,'_per_channel_weight_quantization',side_effect=wrong))
                else:stack.enter_context(patch('flash_provider_source.shutil.disk_usage',return_value=type('Usage',(),{'free':0})()))
                reader=PublicTileReader(scratch,plan,open_projection=source.open_projection,check_active=source.require_tile_source)
                with self.assertRaises((ValueError,InterruptedError)):reader.read(tile)
                self.assertTrue(reader.closed);self.assertEqual(list(scratch.iterdir()),[sibling]);self.assertEqual(source.live_cache_bytes,0)

    def test_closed_checkpoint_rejects_even_an_already_cached_projection(self):
        from public_weights import PublicTileReader
        with ExitStack() as stack:
            _,_,_,plan,_,scratch,source=self.fixture(stack);tile=plan.schedule()[0]
            reader=PublicTileReader(scratch,plan,open_projection=source.open_projection,check_active=source.require_tile_source)
            reader.read(tile);source.close()
            with self.assertRaises(ValueError):reader.read(tile)
            self.assertTrue(reader.closed);self.assertFalse(list(scratch.iterdir()));self.assertEqual(source.live_cache_bytes,0)

    def test_whole_hashes_cannot_be_replaced_by_cached_metadata_or_source_tree_scratch(self):
        from flash_provider_source import ProviderCheckpointSource
        with ExitStack() as stack:
            helper,_,_,plan,_,scratch,_=self.fixture(stack)
            root=helper.provider_source_directory
            with patch('flash_weights.verify_blob') as verify:
                with self.assertRaises(ValueError):ProviderCheckpointSource(root,root,plan)
                verify.assert_not_called()
            path=root/'public.safetensors';raw=path.read_bytes();path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):ProviderCheckpointSource(root,scratch,plan)
            self.assertFalse(list(scratch.iterdir()));self.assertEqual(path.read_bytes(),raw[:-1]+bytes([raw[-1]^1]))

    def test_source_path_and_route_cannot_downgrade_or_expand_legacy_permission(self):
        from checked_bridge import CheckedBridge
        from ssh_route import SSHRoute,validate,ENDPOINT_FIELDS,SSHDeployment
        from test_ssh_route import fixture
        from plan_backends import REBUILD_PROFILE
        with ExitStack() as stack:
            _,_,root,plan,binding,_,_=self.fixture(stack)
            key=Mock()
            with patch('full_bridge.subprocess.Popen') as spawn:
                with self.assertRaises(ValueError):CheckedBridge(root,plan,key,backend_profile=REBUILD_PROFILE,
                    prepared_public_tiles=binding,parameter_profile='glwe-quant12-message35-storage48/v1')
                key.serialize.assert_not_called();spawn.assert_not_called()
            _,value=fixture();endpoint={key:value[key] for key in ENDPOINT_FIELDS};endpoint['remotePublicSource']='/srv/pinned-public-source'
            route=SSHRoute.for_compiled_checkpoint(root,binding,endpoint,('research-provider',))
            self.assertEqual(route.bind(plan,('research-provider',)).definition(),plan.definition())
            with self.assertRaises(ValueError):SSHRoute.for_compiled_checkpoint(root,binding,endpoint,())
            for schema in ('doppler.encrypted-ssh-check-route/v1','doppler.encrypted-ssh-stream-check-route/v1'):
                with self.assertRaises(ValueError):validate(dict(route.definition(),schema=schema))
            with self.assertRaises(ValueError):validate(dict(route.definition(),remotePublicSource='/srv/../unexpected'))

if __name__=='__main__':unittest.main()


class ProviderMetadataTests(unittest.TestCase):
    def test_installed_binding_and_source_subset_sizes_preserve_legacy_geometry(self):
        from flash_provider_source import linear_source_spec
        from flash_text_plan import projection_bindings,GEOMETRY
        from flash_text_rebuild_plan import CACHE_BUDGET_BYTES
        bindings=projection_bindings();source=linear_source_spec()
        self.assertEqual({name:(row['inputWidth'],row['outputWidth']) for name,row in bindings.items()},GEOMETRY)
        self.assertEqual(len(source['shards']),100);self.assertEqual(source['wholeShardBytes'],260799695848)
        self.assertEqual(source['logicalTensorBytes'],250210795520)
        self.assertLessEqual(max(width*height*2 for width,height in GEOMETRY.values()),CACHE_BUDGET_BYTES)
        self.assertEqual(bindings['layers.0.mlp.ops.expert_0_gate']['rows'],(0,640))
        self.assertEqual(bindings['layers.0.mlp.ops.expert_0_up']['rows'],(640,1280))
