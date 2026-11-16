"""Exact first-layer DeltaNet source, geometry and profile separation without native imports."""
import unittest
from flash_delta_plan import GEOMETRY,CONFIG,FlashDeltaPlan
from flash_source_scopes import source_spec
from flash_mlp_plan import FlashMLPPlan
from flash_hyper_plan import FlashHyperPlan
from full_plan import canonical,tiles


class DeltaPlanTests(unittest.TestCase):
    def test_all_five_linear_operations_and_whole_first_shard_are_bound(self):
        source=source_spec('linear-attention')
        self.assertEqual(len(source['shards']),1);self.assertEqual(len(source['tensors']),9)
        self.assertEqual(list(GEOMETRY),CONFIG['projectionOrder'])
        self.assertEqual(GEOMETRY['in_proj_qkv'],(2560,10240));self.assertEqual(GEOMETRY['out_proj'],(6144,2560))
        count=sum(len(tiles(dict(name=name,inputWidth=shape[0],outputWidth=shape[1],weightId='a'*64))) for name,shape in GEOMETRY.items())
        self.assertEqual(count,26);self.assertEqual(count*CONFIG['steps'],104)

    def test_delta_protocol_cannot_substitute_for_mlp_or_hyper(self):
        value=canonical(dict(modelId='a'*64))
        self.assertEqual(len({plan(value).protocol_id for plan in [FlashDeltaPlan,FlashMLPPlan,FlashHyperPlan]}),3)


if __name__=='__main__':unittest.main()
