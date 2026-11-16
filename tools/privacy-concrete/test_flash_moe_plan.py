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

    def test_native_backend_is_explicit_and_cannot_replace_or_mutate_the_stock_policy(self):
        from plan_backends import moe_backend
        from research_parameters import component_parameters
        stock,_,_=moe_backend('flash-dense-moe');native,_,_=moe_backend('flash-dense-moe-ntt')
        self.assertNotEqual(stock['protocol'],native['protocol']);self.assertEqual(stock['profile'],'flash-dense-moe')
        self.assertEqual(component_parameters(stock)['parameters'],component_parameters(native)['parameters'])
        self.assertEqual(component_parameters(native)['evaluationBackend'],'cpu-exact-ntt')
        self.assertFalse(native['productionBackendAvailable']);self.assertEqual(native['maxAbsoluteError'],stock['maxAbsoluteError'])
        native['steps']=99;self.assertEqual(moe_backend('flash-dense-moe-ntt')[0]['steps'],1)
        with self.assertRaises(ValueError):moe_backend('advertised-private-moe')
        with self.assertRaises(ValueError):moe_backend(None)

    def test_component_protocols_cannot_substitute_for_each_other_or_full_generation(self):
        encoded=canonical(dict(modelId='a'*64))
        from flash_moe_ntt_plan import FlashMoENttPlan
        plans=[FullPlan(encoded),FlashExpertPlan(encoded),FlashMoEPlan(encoded),FlashMoENttPlan(encoded)]
        self.assertEqual(len({plan.protocol_id for plan in plans}),4)
        self.assertEqual(len({canonical(plan.frame('session','key',1,'evaluate','tile')) for plan in plans}),4)


if __name__=='__main__':unittest.main()
