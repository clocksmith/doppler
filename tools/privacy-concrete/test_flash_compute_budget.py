import copy
import unittest
from flash_compute_budget import workload,project_native_cost,explain_native_cost


class FlashComputeBudgetTests(unittest.TestCase):
    def cases(self):
        installed=workload()
        return [dict(step=step,rows=5 if step==0 else 1,inputWidth=row['inputWidth'],outputWidth=row['outputWidth'],nativeEvaluateSeconds=1.)
            for step in range(4) for row in installed['shapes']]

    def test_complete_installed_workload_and_explicit_partial_memory_reservation(self):
        value=workload();self.assertEqual(value['projections'],74501);self.assertEqual(value['tilesPerStep'],150868)
        self.assertEqual(len(value['shapes']),15);self.assertEqual(value['logicalCoefficientBytesPerStep'],250210795520)
        self.assertEqual(value['logicalQuantizerMetadataBytes'],1161002304);self.assertEqual(value['largestCoefficientBytes'],1271398400)
        self.assertFalse(value['securityQualified'])

    def test_projection_counts_all_geometries_and_steps_without_claiming_model_latency(self):
        result=project_native_cost(self.cases());self.assertEqual(result['weightedSyntheticNativeEvaluatorSeconds'],603472.)
        self.assertFalse(result['measuredFullModelLatency']);self.assertFalse(result['productionBackendAvailable'])
        for failure in ('missing','duplicate','shape','rows','nonfinite','negative'):
            with self.subTest(failure=failure):
                rows=copy.deepcopy(self.cases())
                if failure=='missing':rows.pop()
                if failure=='duplicate':rows[-1]=rows[0]
                if failure=='shape':rows[0]['inputWidth']=1
                if failure=='rows':rows[0]['rows']=1
                if failure=='nonfinite':rows[0]['nativeEvaluateSeconds']=float('nan')
                if failure=='negative':rows[0]['nativeEvaluateSeconds']=-1
                with self.assertRaises(ValueError):project_native_cost(rows)

    def test_scope_is_four_outputs_with_five_row_prefill_and_three_decode_steps(self):
        cases=self.cases()
        for row in cases:row['nativeEvaluateSeconds']=5. if row['step']==0 else 1.
        result=explain_native_cost(cases)
        self.assertEqual(result['requestScope'],dict(outputTokens=4,prefillInputRows=5,singleRowDecodeSteps=3,totalTiles=603472))
        self.assertEqual([row['inputRows'] for row in result['steps']],[5,1,1,1])
        self.assertEqual([row['weightedSyntheticNativeEvaluatorSeconds'] for row in result['steps']],[754340.,150868.,150868.,150868.])
        self.assertEqual(sum(row['weightedSyntheticNativeEvaluatorSeconds'] for row in result['steps']),result['projection']['weightedSyntheticNativeEvaluatorSeconds'])
        self.assertEqual(sum(row['tilesPerStep'] for row in result['projectionFamilies']),150868)
        self.assertEqual(sum(row['weightedSyntheticNativeEvaluatorSeconds'] for row in result['projectionFamilies']),result['projection']['weightedSyntheticNativeEvaluatorSeconds'])
        experts=next(row for row in result['projectionFamilies'] if row['family']=='all-private-routed-experts')
        self.assertEqual(experts['tilesPerStep'],147456);self.assertFalse(result['measuredFullModelLatency'])
        with self.assertRaises(ValueError):explain_native_cost(cases[:-1])


if __name__=='__main__':unittest.main()
