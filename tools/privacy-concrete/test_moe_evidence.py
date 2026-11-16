import json
from pathlib import Path
import tempfile
import unittest
from collect_moe_evidence import collect,SCHEDULE,METRICS
from flash_moe_plan import CONFIG,PROGRAM,SOURCE_ID


class MoEEvidenceTests(unittest.TestCase):
    def test_partial_or_cancelled_route_retains_only_fixed_public_progress(self):
        self.assertEqual(len(SCHEDULE),2058)
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'report';progress=Path(directory)/'progress'
            progress.write_text('private-canary\n'+json.dumps(dict(event='verified-tile',sequence=1,projection=SCHEDULE[0],hidden='private-canary'))+'\n')
            value=collect(report,progress,'public-revision');self.assertFalse(value['completed']);self.assertEqual(value['verifiedTilesObserved'],1)
            self.assertNotIn('private-canary',json.dumps(value))
            report.write_text(json.dumps(dict(status=dict(event='run-failed',failureCode='cancelled',message='private-canary'))))
            value=collect(report,progress,'public-revision');self.assertEqual(value['failureCode'],'cancelled');self.assertNotIn('private-canary',json.dumps(value))

    def test_complete_public_route_is_projected_and_invalid_claims_never_complete(self):
        report_value=dict(schema='doppler.actual-flash-dense-moe-execution/v1',fixtureId='flash-dense-moe-sine-2560',productionBackendAvailable=False,
            securityQualified=False,fullFlashModelExecuted=False,sourceId=SOURCE_ID,programId=PROGRAM.identity,protocol=CONFIG['protocol'],expertCount=512,
            topK=10,logicalProjections=1541,linearModules=1029,plannedEncryptedTiles=2058,encryptedTiles=2058,allExpertsVisited=True,
            execution='encrypted',encryptedExecutionCompleted=True,quantizedReferenceCheckPassed=True,modelId='a'*64,planId='b'*64,
            private='private-canary',**{key:1 for key in METRICS})
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'report';progress=Path(directory)/'progress'
            progress.write_text(''.join(json.dumps(dict(event='verified-tile',sequence=i+1,projection=name))+'\n' for i,name in enumerate(SCHEDULE)))
            report.write_text(json.dumps(report_value));value=collect(report,progress,'public-revision');self.assertTrue(value['completed']);self.assertNotIn('private-canary',json.dumps(value))
            for change in [dict(securityQualified=True),dict(fullFlashModelExecuted=True),dict(encryptedTiles=2057),dict(expertCount=True),dict(executeMs=float('nan')),dict(fixtureId='private-canary')]:
                report.write_text(json.dumps(dict(report_value,**change)));self.assertFalse(collect(report,progress,'public-revision')['completed'])
            report.write_text(json.dumps(report_value));progress.write_text(json.dumps(dict(event='verified-tile',sequence=2,projection=SCHEDULE[0])))
            value=collect(report,progress,'public-revision');self.assertFalse(value['completed']);self.assertFalse(value['progressValid'])
            progress.write_text('');self.assertFalse(collect(report,progress,'public-revision')['completed'])


if __name__=='__main__':unittest.main()
