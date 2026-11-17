import copy
import unittest
from flash_compute_budget import workload,project_native_cost


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


if __name__=='__main__':unittest.main()
