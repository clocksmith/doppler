"""Request ownership, terminal cancellation and loader rejection without importing Torch."""
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from flash_delta_state import DeltaContext,DeltaRequestState
from flash_reference import load_reference


class DeltaOwnershipTests(unittest.TestCase):
    def context(self):return DeltaContext('tenant','privacy','model','key','request')

    def test_namespace_or_execution_change_clears_state_and_prevents_retry(self):
        context=self.context()
        for changed in [replace(context,tenant_id='other'),replace(context,key_id='other'),replace(context,privacy_id='other'),replace(context,model_id='other'),replace(context,request_id='other')]:
            state=DeltaRequestState(context);state.conv[(0,0)]=object()
            with self.assertRaises(ValueError):state.begin(changed,('project',),None,'public-reference')
            self.assertTrue(state.failed);self.assertEqual(state.conv,{})
            with self.assertRaises(ValueError):state.begin(context,('project',),None,'public-reference')
        state=DeltaRequestState(context);state.begin(context,('project',),None,'verified-projections')
        state.projection('project');state.verified();state.complete(3)
        with self.assertRaises(ValueError):state.begin(context,('project',),None,'public-reference')

    def test_cancellation_and_incomplete_routes_are_terminal(self):
        event=threading.Event();event.set();state=DeltaRequestState(self.context())
        with self.assertRaises(InterruptedError):state.begin(self.context(),('project',),event,'verified-projections')
        event.clear()
        with self.assertRaises(ValueError):state.begin(self.context(),('project',),event,'verified-projections')
        state=DeltaRequestState(self.context());state.begin(self.context(),('first','last'),None,'public-reference')
        with self.assertRaises(ValueError):state.complete(1)
        self.assertTrue(state.failed)

    def test_unknown_or_changed_source_never_executes_downloaded_python(self):
        with self.assertRaises(ValueError):load_reference('/does-not-exist','advertised-unreviewed-program')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.py';path.write_text('raise RuntimeError("untrusted code executed")')
            with self.assertRaises(ValueError):load_reference(path,'delta')


if __name__=='__main__':unittest.main()
