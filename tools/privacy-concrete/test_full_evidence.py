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
            value=dict(publicFixtureId='qwen05b-capital',qualified=False,productionBackendAvailable=False,
                       linearProjectionCount=169,encryptedSteps=4,encryptedTileCount=1548,
                       matchesFullQuantizedGreedyReference=True,clientLocalOutputText='public fixture output',
                       clientLocalOutputTokens=[1,2,3,4],descriptor={'secretKey':'private-canary'},
                       implementationVersions={'torch':'2.3.1+cpu','private':'private-canary'},integrity='private-canary')
            report.write_text(json.dumps(value))
            result=collect(report,progress,'public-revision')
            self.assertTrue(result['completed'])
            self.assertNotIn('private-canary',json.dumps(result))

    def test_private_or_unqualified_reports_cannot_publish_text(self):
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'result.json';progress=Path(directory)/'progress.jsonl'
            report.write_text(json.dumps(dict(publicFixtureId='private',clientLocalOutputText='private-output-canary')))
            value=collect(report,progress,'public-revision')
            self.assertFalse(value['completed'])
            self.assertNotIn('private-output-canary',json.dumps(value))


if __name__=='__main__':unittest.main()
