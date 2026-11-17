"""Real quantizer/safetensor fixture identity across exec; no encryption."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in real quantizer and exec fixture')
class NativeProcessFixtureTests(unittest.TestCase):
    def test_independent_processes_construct_identical_public_sources_and_coefficients(self):
        from native_process_fixture import fixture
        from protocol import digest
        from full_plan import canonical
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            with fixture(root/'client') as scope:
                expected=dict(sourceId=digest(canonical(scope.stream.linear_source_spec())),rows=scope.rows)
            code='''import json,sys
from native_process_fixture import fixture
from protocol import digest
from full_plan import canonical
with fixture(sys.argv[1]) as scope:
 print(json.dumps(dict(sourceId=digest(canonical(scope.stream.linear_source_spec())),rows=scope.rows)))
'''
            environment=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parent),HF_HOME=str(root/'hf'))
            raw=subprocess.check_output([sys.executable,'-c',code,str(root/'provider')],env=environment,timeout=60)
            self.assertEqual(json.loads(raw),expected)


if __name__=='__main__':unittest.main()
