"""Adversarial fixtures for the separate prefill comparator, including python -O."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('comparison', Path(__file__).with_name('compare-f32kv-reference.py'))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparatorContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sha = lambda b: hashlib.sha256(b).hexdigest()
        self.inputs = self.write('inputs.json', dict(sequences=[dict(id='short', tokens=[1, 2])]))
        self.manifest = self.write('manifest.json', dict(architecture=dict(vocabSize=4)))
        self.producer = self.root/'producer.py'
        self.producer.write_text('# A pinned fixture producer, not a model reference\n')
        self.values = np.array([0.1, 0.2, 0.3, 0.4], dtype='<f4')
        self.raw = self.root/'gpu.f32'
        self.raw.write_bytes(self.values.tobytes())
        self.source = self.root/'source.npy'
        np.save(self.source, self.values.reshape(1, 4), allow_pickle=False)
        self.reference = dict(schema='doppler.qwen38-learned-forward-reference/v1', dtype='float32',
            producerSha256=self.sha(self.producer.read_bytes()), artifactManifestSha256=self.sha(self.manifest.read_bytes()),
            inputsSha256=self.sha(self.inputs.read_bytes()), sequences=dict(short=dict(tokens=2, samplePositions=[1],
                logitsFile=str(self.source), logitsSha256=self.sha(self.source.read_bytes()), argmax=[3, 3])))
        self.binding = dict(kvStorageDtype='f32', argv=[str(self.producer), '--inputs', str(self.inputs), '--artifact', str(self.root)],
            producerSha256=self.reference['producerSha256'], manifestSha256=self.reference['artifactManifestSha256'],
            inputsSha256=self.reference['inputsSha256'])
        self.gpu = dict(completed=True, gpuErrors=[], cleanupFailures=[], manifestSha256=self.binding['manifestSha256'],
            inputsSha256=self.binding['inputsSha256'], runtimeConfig=dict(inference=dict(session=dict(kvcache=dict(kvDtype='f32')))),
            rows=[dict(id='short', inputTokenCount=2, kvStorageDtype='f32', prefillLogitsFile=str(self.raw),
                prefillLogitsSha256=self.sha(self.raw.read_bytes()), tokenId=3)])

    def write(self, name, value):
        file = self.root/name
        file.write_text(json.dumps(value))
        return file

    def run_comparison(self):
        args = [self.write('gpu.json', self.gpu), self.write('reference.json', self.reference),
                self.write('binding.json', self.binding), self.root/'comparison.json']
        with contextlib.redirect_stdout(io.StringIO()):
            comparison.main(*args)
        return json.loads(args[-1].read_text())

    def test_exact_rows_pass_only_their_named_scope(self):
        report = self.run_comparison()
        self.assertTrue(report['passed'])
        self.assertFalse(report['fullModelQualified'])
        self.assertEqual(report['rows'][0]['outsidePolicy'], 0)

    def test_agreeing_top_token_cannot_hide_bad_tail(self):
        values = self.values.copy()
        values[0] -= 1
        self.raw.write_bytes(values.tobytes())
        self.gpu['rows'][0]['prefillLogitsSha256'] = self.sha(self.raw.read_bytes())
        report = self.run_comparison()
        self.assertTrue(report['rows'][0]['firstTokenAgrees'])
        self.assertFalse(report['passed'])
        self.assertEqual(report['rows'][0]['outsidePolicy'], 1)

    def test_truncation_is_not_partial_success(self):
        self.raw.write_bytes(self.values[:3].tobytes())
        self.gpu['rows'][0]['prefillLogitsSha256'] = self.sha(self.raw.read_bytes())
        with self.assertRaises(ValueError): self.run_comparison()

    def test_configured_dtype_cannot_replace_observed_dtype(self):
        self.gpu['rows'][0]['kvStorageDtype'] = 'f16'
        with self.assertRaises(ValueError): self.run_comparison()

    def test_replacement_routing_is_not_acceptance(self):
        self.gpu['routingControl'] = dict(replaced=[0])
        with self.assertRaises(ValueError): self.run_comparison()

    def test_reference_identity_mismatch_rejects(self):
        self.binding['inputsSha256'] = '0'*64
        with self.assertRaises(ValueError): self.run_comparison()


if __name__ == '__main__':
    unittest.main()
