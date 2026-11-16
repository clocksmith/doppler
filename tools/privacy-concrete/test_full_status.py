import json
from pathlib import Path
import tempfile
import unittest
from full_status import ExecutionRejected, failure_event
from collect_full_evidence import collect


class FullStatusTests(unittest.TestCase):
    def test_exception_messages_and_custom_types_never_enter_public_status(self):
        class PrivatePromptCanary(Exception):pass
        for error,expected in [(InterruptedError('private-canary'),'cancelled'),
                               (TimeoutError('private-canary'),'deadline-exceeded'),
                               (MemoryError('private-canary'),'resource-exhausted'),
                               (ValueError('private-canary'),'contract-rejected'),
                               (PrivatePromptCanary('private-canary'),'execution-failed'),
                               (ExecutionRejected('token-reference-mismatch'),'token-reference-mismatch')]:
            event=failure_event(error,123)
            self.assertEqual(event,dict(event='run-failed',failureCode=expected,elapsedMs=123))
            self.assertNotIn('canary',json.dumps(event))
        with self.assertRaises(ValueError):ExecutionRejected('private-canary')
        for elapsed in [float('nan'),float('inf'),True,-1]:
            with self.assertRaises(ValueError):failure_event(ValueError(),elapsed)

    def test_collector_publishes_only_known_failure_code(self):
        with tempfile.TemporaryDirectory() as directory:
            progress=Path(directory)/'progress.jsonl';report=Path(directory)/'report.json'
            progress.write_text(json.dumps(dict(event='run-failed',failureCode='deadline-exceeded',error='private-canary'))+'\n')
            result=collect(report,progress,'revision')
            self.assertEqual(result['failureCode'],'deadline-exceeded')
            self.assertFalse(result['completed']);self.assertNotIn('private-canary',json.dumps(result))
            progress.write_text(json.dumps(dict(event='run-failed',failureCode='private-canary'))+'\n')
            self.assertNotIn('failureCode',collect(report,progress,'revision'))
            progress.write_text(json.dumps(dict(event='run-failed',failureCode=['private-canary']))+'\n')
            report.write_text('[]')
            self.assertFalse(collect(report,progress,'revision')['completed'])


if __name__=='__main__':unittest.main()
