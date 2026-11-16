"""Pure source-scope, geometry and protocol boundaries; no native crypto/model workload."""
import unittest
from flash_source_scopes import source_spec,BASE
from flash_hyper_program import flash_hyper_program,require_source_config
from flash_hyper_plan import FlashHyperPlan
from flash_expert_plan import FlashExpertPlan
from flash_moe_plan import FlashMoEPlan
from flash_mlp_plan import FlashMLPPlan,OPERATIONS
from full_plan import canonical,tiles
from protocol import digest


class HyperPlanTests(unittest.TestCase):
    def test_installed_source_scopes_preserve_moe_and_bind_new_hyper_tensors(self):
        self.assertEqual(canonical(source_spec('moe')),canonical(BASE))
        hyper=source_spec('mlp-hyper');block=source_spec('mlp-block')
        self.assertEqual(len(hyper['tensors']),4);self.assertEqual(len(hyper['shards']),1)
        self.assertEqual(len(block['tensors']),11);self.assertEqual(len(block['shards']),2)
        self.assertEqual(len({digest(canonical(spec)) for spec in [BASE,hyper,block]}),3)
        hyper['shards'].clear()
        self.assertEqual(len(source_spec('mlp-hyper')['shards']),1)
        with self.assertRaises(ValueError):source_spec('advertised-unreviewed-flash-source')

    def test_source_geometry_and_epsilon_are_required_before_graph_execution(self):
        program=flash_hyper_program()
        config=dict(hc_count=4,hidden_size=2560,hc_lowrank=320,rms_norm_eps=1e-6)
        require_source_config(program,config)
        for change in [dict(hc_count=True),dict(hc_lowrank=640),dict(rms_norm_eps=1e-5)]:
            with self.assertRaises(ValueError):require_source_config(program,dict(config,**change))
        operations=program.contract()['operations']
        count=sum(len(tiles(dict(name=row['id'],inputWidth=row['inputWidth'],outputWidth=row['outputWidth'],weightId='a'*64))) for row in operations)
        self.assertEqual(count,15)

    def test_complete_mlp_route_includes_all_hyper_and_moe_operations(self):
        self.assertEqual(len(OPERATIONS),1032)
        self.assertEqual([row['name'] for row in OPERATIONS[:3]],['hyper/down','hyper/up','hyper/inject'])
        self.assertEqual(OPERATIONS[3]['name'],'shared/gate')
        self.assertEqual(sum(len(tiles(dict(row,weightId='a'*64))) for row in OPERATIONS),2073)

    def test_hyper_feedback_protocol_does_not_admit_other_flash_components(self):
        raw=canonical(dict(modelId='a'*64));plans=[FlashHyperPlan(raw),FlashExpertPlan(raw),FlashMoEPlan(raw),FlashMLPPlan(raw)]
        self.assertEqual(len({plan.protocol_id for plan in plans}),4)


if __name__=='__main__':unittest.main()
