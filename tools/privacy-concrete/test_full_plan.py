import json
import unittest
from unittest.mock import patch
from full_plan import FullPlan, canonical, ranges, tiles
from protocol import check


class FullPlanTests(unittest.TestCase):
    def test_wide_dots_are_covered_once_by_admitted_tiles(self):
        projection=dict(name='wide',weightId='a'*64,inputWidth=4864,outputWidth=896)
        layout=tiles(projection)
        self.assertEqual([(row['inputBegin'],row['inputEnd']) for row in layout],[(0,2048),(2048,4096),(4096,4864)])
        self.assertEqual(len(set(row['id'] for row in layout)),3)
        self.assertEqual(sum(row['inputEnd']-row['inputBegin'] for row in layout),4864)

    def test_large_vocabulary_output_is_tiled_and_bound_to_each_submatrix(self):
        row=dict(name='lm_head',weightId='b'*64,inputWidth=896,outputWidth=151936)
        layout=tiles(row)
        self.assertEqual(len(layout),75)
        self.assertEqual(layout[-1]['outputEnd'],151936)
        self.assertEqual(layout[-1]['outputEnd']-layout[-1]['outputBegin'],384)
        self.assertNotEqual(layout[0]['id'],layout[1]['id'])
        with self.assertRaises(ValueError):tiles(row,input_limit=4096,output_limit=0)

    def test_reply_cannot_change_tile_or_plan_identity(self):
        plan=FullPlan(canonical(dict(modelId='c'*64)))
        expected=plan.frame('s','k',1,'result','tile-one')
        for changed in [plan.frame('s','k',1,'result','tile-two'),plan.frame('s','other',1,'result','tile-one')]:
            with self.assertRaises(ValueError):check(changed,expected)

    def test_many_frames_keep_the_exact_binding_without_reparsing_the_full_model(self):
        original=dict(modelId='c'*64,projections=[dict(name=str(index)) for index in range(2000)])
        plan=FullPlan(canonical(original));expected_id=plan.identity
        with patch.object(FullPlan,'definition',autospec=True,side_effect=lambda self:json.loads(self.encoded)) as definition:
            first=plan.frame('s','k',1,'evaluate','tile')
            for index in range(2,1002):
                frame=plan.frame('s','k',index,'evaluate','tile')
                self.assertEqual(frame,dict(first,sequence=index))
            self.assertEqual(definition.call_count,1)
        changed=plan.definition();changed['modelId']='d'*64;changed['projections'].clear()
        original['modelId']='e'*64
        self.assertEqual(plan.frame('s','k',1,'evaluate','tile'),first)
        self.assertEqual(plan.identity,expected_id);self.assertEqual(len(plan.definition()['projections']),2000)

    def test_cpu_torch_variant_is_explicit_and_unknown_variants_fail(self):
        from full_plan import implementation_versions
        from protocol import POLICY
        def installed(dependency):
            return '2.3.1+cpu' if dependency=='torch' else POLICY['dependencies'][dependency]
        with patch('full_plan.version',side_effect=installed):
            self.assertEqual(implementation_versions()['torch'],'2.3.1+cpu')
        with patch('full_plan.version',return_value='2.3.1+unreviewed'):
            with self.assertRaises(ValueError):implementation_versions()

    def test_invalid_geometry_has_no_fallback(self):
        for width in [0,-1,True]:
            with self.assertRaises(ValueError):ranges(width,2048)
        with self.assertRaises(ValueError):tiles(dict(name='x',weightId='x',inputWidth=4864,outputWidth=1),input_limit=4096)


if __name__=='__main__':unittest.main()
