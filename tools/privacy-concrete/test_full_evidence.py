import json
from pathlib import Path
import tempfile
import unittest
from collect_full_evidence import collect


class FullEvidenceTests(unittest.TestCase):
    def test_incomplete_logs_are_projected_without_private_or_unstructured_data(self):
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'result.json';progress=Path(directory)/'progress.jsonl'
            progress.write_text('private-input-canary\n'+json.dumps(dict(event='verified-tile',sequence=1,projection='lm_head',secretKey='private-input-canary'))+'\n')
            value=collect(report,progress,'public-revision')
            self.assertFalse(value['completed'])
            self.assertEqual(value['verifiedTilesObserved'],1)
            self.assertNotIn('private-input-canary',json.dumps(value))

    def test_fixed_fixture_evidence_drops_nested_arbitrary_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'result.json';progress=Path(directory)/'progress.jsonl'
            value=dict(schema='doppler.full-qwen-hybrid-execution/v1',planId='a'*64,modelId='b'*64,publicFixtureId='qwen05b-capital',qualified=False,productionBackendAvailable=False,
                       linearProjectionCount=169,encryptedSteps=4,encryptedTileCount=1548,
                       matchesFullQuantizedGreedyReference=True,clientLocalOutputText='public fixture output',
                       clientLocalOutputTokens=[1,2,3,4],executionRouteId='c'*64,transport='ssh-ed25519-pinned',descriptor={'secretKey':'private-canary'},
                       implementationVersions={'torch':'2.3.1+cpu','private':'private-canary'},integrity='private-canary')
            report.write_text(json.dumps(value))
            result=collect(report,progress,'public-revision')
            self.assertTrue(result['completed'])
            self.assertEqual(result['report']['executionRouteId'],'c'*64)
            self.assertEqual(result['report']['transport'],'ssh-ed25519-pinned')
            self.assertNotIn('private-canary',json.dumps(result))
            for change in [dict(clientLocalOutputText={'private':'private-canary'}),dict(clientLocalOutputTokens=[True,2,3,4]),dict(schema='unknown'),dict(planId='unknown')]:
                report.write_text(json.dumps(dict(value,**change)))
                rejected=collect(report,progress,'public-revision')
                self.assertFalse(rejected['completed']);self.assertNotIn('private-canary',json.dumps(rejected))
            fast=dict(value,backendProfile='qwen-all-linear-ntt',protocol='concrete-glwe-qwen-all-linear-ntt-replay/v1',implementationVersions={'concrete-ml-extensions':'0.1.9+ntt.1','native-source-id':'d'*64,'native-binary-sha256':'e'*64,'private':'private-canary'})
            report.write_text(json.dumps(fast));accepted=collect(report,progress,'public-revision')
            self.assertTrue(accepted['completed']);self.assertEqual(accepted['report']['backendProfile'],'qwen-all-linear-ntt')
            self.assertEqual(len(accepted['report']['nativeImplementation']),2);self.assertNotIn('private-canary',json.dumps(accepted))
            for change in [dict(protocol='unknown'),dict(backendProfile={}),dict(protocol=[]),dict(implementationVersions={})]:
                report.write_text(json.dumps(dict(fast,**change)));self.assertFalse(collect(report,progress,'public-revision')['completed'])
            high=dict(fast,backendProfile='qwen-all-linear-ntt-48',protocol='concrete-glwe-qwen-all-linear-ntt-48-replay/v1')
            report.write_text(json.dumps(high));self.assertTrue(collect(report,progress,'public-revision')['completed'])
            report.write_text(json.dumps(value));progress.write_text(json.dumps(dict(event='run-failed',failureCode='cancelled'))+'\n')
            self.assertFalse(collect(report,progress,'public-revision')['completed'])

    def test_private_or_unqualified_reports_cannot_publish_text(self):
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'result.json';progress=Path(directory)/'progress.jsonl'
            report.write_text(json.dumps(dict(publicFixtureId='private',clientLocalOutputText='private-output-canary')))
            value=collect(report,progress,'public-revision')
            self.assertFalse(value['completed'])
            self.assertNotIn('private-output-canary',json.dumps(value))


if __name__=='__main__':unittest.main()
