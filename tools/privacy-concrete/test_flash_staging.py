"""Whole-object staged materialization: exact coverage, isolation and terminal cleanup."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from flash_source_scopes import TEXT,BASE,source_spec,staging_spec
from flash_stage_source import staged_source
from flash_storage_plan import storage_plan,TABLE
from full_plan import canonical
from protocol import digest


def fixture():
    name='model.language_model.layers.0.mlp.shared_expert.gate_proj.weight'
    header=canonical({name:dict(dtype='BF16',shape=[2,2],data_offsets=[0,8])})
    header+=b' '*((-len(header))%8)
    blob=struct.pack('<Q',len(header))+header+struct.pack('<4H',0x3f80,0x4000,0x4040,0x4080)
    files={'config.json':b'{"text_config":{}}','model.safetensors.index.json':canonical(dict(weight_map={name:'public.safetensors'})),'public.safetensors':blob}
    def record(data):return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    spec=dict(schema='doppler.flash-staged-model-part-weights/v1',part='decoder',layerIndex=0,repository='public/fixture',revision='fixed',
              files={n:record(data) for n,data in files.items() if n!='public.safetensors'},shards={'public.safetensors':record(blob)},
              tensors={name:dict(shard='public.safetensors',shape=[2,2],dtype='BF16',dataOffsets=[0,8])})
    return name,spec,files


class FlashStageTests(unittest.TestCase):
    def test_all_installed_parts_cover_every_checkpoint_tensor_without_expanding_legacy_scope(self):
        parts=[staging_spec('decoder',i) for i in range(TEXT['layerCount'])]+[staging_spec('output')]
        covered={name for spec in parts for name in spec['tensors']}
        table_names={row['name'] for row in TABLE['parts']}
        self.assertFalse(covered&table_names)
        self.assertEqual(covered|table_names|{'model.language_model.embed_tokens.weight'},set(TEXT['tensors']))
        for index,spec in enumerate(parts[:-1]):
            self.assertTrue(all(name.startswith(f'model.language_model.layers.{index}.') for name in spec['tensors']))
            self.assertEqual(sum('.mlp.experts.' in name for name in spec['tensors']),2)
            self.assertEqual(spec['parentSourceId'],digest(canonical(TEXT)));self.assertFalse(spec['completeTextModel'])
        self.assertEqual(set(parts[1]['separatelyOwnedLocalTableTensors']),table_names)
        self.assertEqual(source_spec('moe'),BASE)
        parts[0]['tensors'].clear();self.assertTrue(staging_spec('decoder',0)['tensors'])

    def test_invalid_stage_rejects_before_filesystem_or_network_and_is_not_a_backend(self):
        for part,index in [('decoder',True),('decoder',48),('decoder',-1),('output',0),('advertised-private',None)]:
            with patch('flash_stage_source.tempfile.TemporaryDirectory',side_effect=AssertionError('filesystem before admission')),patch('flash_stage_source.provision_spec') as acquire:
                with self.assertRaises(ValueError):
                    with staged_source('/unused',part,index):self.fail('unsupported stage entered')
                acquire.assert_not_called()
        from plan_backends import load_backend_plan
        with self.assertRaises(ValueError):load_backend_plan('/unused','flash-staged-full-model')

    def test_whole_verified_stage_closes_and_removes_only_owned_files_on_success_or_failure(self):
        name,spec,files=fixture()
        for fail in [False,True]:
            with tempfile.TemporaryDirectory() as parent:
                sibling=Path(parent)/'unrelated';sibling.write_text('preserve');captured=[]
                def acquire(directory,unused):
                    captured.append(Path(directory))
                    for file,data in files.items():(Path(directory)/file).write_bytes(data)
                with patch('flash_stage_source.staging_spec',return_value=spec),patch('flash_stage_source.provision_spec',side_effect=acquire):
                    try:
                        with staged_source(parent,'decoder',0) as source:
                            self.assertEqual(source.headers['public.safetensors'][name]['shape'],[2,2])
                            self.assertFalse(source.closed)
                            if fail:raise RuntimeError('caller failure')
                    except RuntimeError:
                        if not fail:raise
                self.assertTrue(source.closed);self.assertFalse(captured[0].exists());self.assertEqual(sibling.read_text(),'preserve')
                with self.assertRaises(ValueError):source.tensor(name)

    def test_corrupt_stage_is_never_yielded_and_is_removed_without_touching_other_files(self):
        name,spec,files=fixture()
        with tempfile.TemporaryDirectory() as parent:
            captured=[]
            def acquire(directory,unused):
                captured.append(Path(directory))
                for file,data in files.items():(Path(directory)/file).write_bytes(data)
                (Path(directory)/'public.safetensors').write_bytes(b'corrupt')
            with patch('flash_stage_source.staging_spec',return_value=spec),patch('flash_stage_source.provision_spec',side_effect=acquire),self.assertRaises(ValueError):
                with staged_source(parent,'decoder',0):self.fail('corrupt public material admitted')
            self.assertFalse(captured[0].exists())

    def test_storage_bounds_account_for_persistent_rows_and_distinct_coefficient_widths(self):
        narrow=storage_plan(8);wide=storage_plan(12)
        self.assertEqual(len(narrow['stages']),TEXT['layerCount']+1)
        self.assertEqual(wide['allPartCoefficientCacheBytesUpperBound'],2*narrow['allPartCoefficientCacheBytesUpperBound'])
        self.assertEqual(wide['stagedRawSourcePeakBytes'],narrow['stagedRawSourcePeakBytes'])
        self.assertLess(wide['stagedRawAndOneCachePeakBytesUpperBound'],wide['fullyResidentSourceAndCachesBytesUpperBound'])
        self.assertGreater(wide['stagedRawSourcePeakBytes'],wide['persistentLocalRowStoreWholeBytes'])
        self.assertFalse(wide['downloadsPerformed']);self.assertFalse(wide['productionBackendAvailable'])
        for bits in [True,4,16]:
            with self.assertRaises(ValueError):storage_plan(bits)


if __name__=='__main__':unittest.main()
