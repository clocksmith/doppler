"""The complete fixed public lowering is independent of private top-k choices."""
import unittest
from flash_moe_plan import CONFIG,PROGRAM,FlashMoEPlan,lowered_operations
from flash_expert_plan import FlashExpertPlan
from full_plan import FullPlan,canonical,tiles


class FlashMoEPlanTests(unittest.TestCase):
    def test_all_512_experts_and_shared_router_are_in_the_fixed_lowered_route(self):
        operations=lowered_operations()
        self.assertEqual(len(PROGRAM.contract()['operations']),1541)
        self.assertEqual(len(operations),1029)
        self.assertEqual([row['name'] for row in operations[:4]],['shared/gate','shared/up','shared/down','router/logits'])
        self.assertEqual(operations[-1],dict(name='shared/mix',inputWidth=2560,outputWidth=1))
        self.assertEqual([row['name'] for row in operations[4:-1]],
                         [f'expert/{index}/{leaf}' for index in range(512) for leaf in ['gate_up','down']])
        count=sum(len(tiles(dict(row,weightId='a'*64),CONFIG['inputTileWidth'],CONFIG['outputTileWidth'])) for row in operations)
        self.assertEqual(count,2058)

    def test_component_protocols_cannot_substitute_for_each_other_or_full_generation(self):
        encoded=canonical(dict(modelId='a'*64))
        plans=[FullPlan(encoded),FlashExpertPlan(encoded),FlashMoEPlan(encoded)]
        self.assertEqual(len({plan.protocol_id for plan in plans}),3)
        self.assertEqual(len({canonical(plan.frame('session','key',1,'evaluate','tile')) for plan in plans}),3)


if __name__=='__main__':unittest.main()
