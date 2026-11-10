"""The learned-model comparator must reject wrong evidence and preserve real failures."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'artifacts/qwen38-flash-next-20261013/compare-learned-reference.py'
POLICY = ROOT / 'src/config/evidence/boundary-evidence-policy.json'
POLICY_ID = 'doppler.boundary-tolerance/quantized-v1'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


class LearnedComparison(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {name: self.root / (name + '.json') for name in ['inputs', 'decode-inputs', 'reference', 'decode', 'gpu', 'protocol']}
        self.inputs = dict(sequences=[dict(id='short', tokens=[0, 1])])
        self.decode_inputs = dict(sequences=[dict(id='short-next', tokens=[0, 1, 2]), dict(id='short-next2', tokens=[0, 1, 2, 1])])
        self.save('inputs', self.inputs)
        self.save('decode-inputs', self.decode_inputs)
        self.values = [np.array([-1, 0, 3, 0], dtype='<f4'), np.array([0, 2, 0, 0], dtype='<f4'), np.array([0, 0, 0, 3], dtype='<f4')]
        self.source_cases = {}
        for step, name in enumerate(['short', 'short-next', 'short-next2']):
            file = self.root / (name + '.npy')
            np.save(file, self.values[step][None, :], allow_pickle=False)
            length = 2 + step
            self.source_cases[name] = dict(tokens=length, samplePositions=[length - 1],
                                          argmax=[0] * (length - 1) + [int(self.values[step].argmax())],
                                          logitsFile=file.name, logitsSha256=sha(file.read_bytes()), layers={})
        self.reference = dict(artifactManifestSha256='a' * 64, inputs='inputs.json',
                              inputsSha256=sha(self.paths['inputs'].read_bytes()), sequences={'short': self.source_cases['short']})
        self.decode = dict(artifactManifestSha256='a' * 64, inputsSha256=sha(self.paths['decode-inputs'].read_bytes()),
                           sequences={name: self.source_cases[name] for name in ['short-next', 'short-next2']})
        self.save('reference', self.reference)
        self.save('decode', self.decode)
        frames = []
        for step, values in enumerate(self.values):
            file = self.root / f'step{step}.f32'
            file.write_bytes(values.tobytes())
            frames.append(dict(step=step, file=str(file), sha256=sha(file.read_bytes()), shape=[4]))
        self.gpu = dict(completed=True, gpuErrors=[], cleanupFailures=[], manifestSha256='a' * 64,
                        inputsSha256=self.reference['inputsSha256'], referenceSha256=sha(self.paths['reference'].read_bytes()), runSha256='b' * 64,
                        rows=[dict(id='short', inputTokenCount=2, tokenId=2, tokenIds=[2, 1, 3], kvStorageDtype='f32',
                                   prefillLogitsFile=frames[0]['file'], prefillLogitsSha256=frames[0]['sha256'], logitFrames=frames)])
        self.protocol = dict(schema='doppler.flash-next-learned-decode-protocol/v1',
                             inputsSha256=self.reference['inputsSha256'], manifestSha256='a' * 64,
                             originalReferenceSha256=sha(self.paths['reference'].read_bytes()),
                             decodeReferenceSha256=sha(self.paths['decode'].read_bytes()),
                             decodeInputsSha256=sha(self.paths['decode-inputs'].read_bytes()),
                             policySha256=sha(POLICY.read_bytes()), policyId=POLICY_ID,
                             tolerancePolicy=json.loads(POLICY.read_bytes())['tolerancePolicies'][POLICY_ID], runSha256='b' * 64,
                             sourceGreedyChainVerifiedBeforeGpu=True, maximumGenerationRowsPerPrompt=3,
                             chains=[dict(id='short', expectedTokenIds=[2, 1, 3])])
        self.save('gpu', self.gpu)
        self.save('protocol', self.protocol)

    def save(self, name, value):
        self.paths[name].write_text(json.dumps(value))

    def compare(self, optimized=False, legacy=False):
        output = self.root / ('result-' + str(optimized) + '.json')
        command = [sys.executable] + (['-O'] if optimized else []) + [str(SCRIPT), str(self.paths['gpu']), str(self.paths['reference']),
                    str(output), str(self.paths['decode']), str(self.paths['decode-inputs']), '--inputs', str(self.paths['inputs']), '--policy', str(POLICY)]
        if not legacy:
            command += ['--kv-dtype', 'f32', '--protocol', str(self.paths['protocol'])]
        process = subprocess.run(command, capture_output=True, text=True)
        return process, json.loads(output.read_bytes()) if output.exists() else None

    def test_prefill_and_decode_check_all_vectors_and_greedy_choices(self):
        process, result = self.compare()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(result['passed'])
        self.assertEqual(result['independentDecodeRowsChecked'], 2)
        self.assertTrue(result['rows'][0]['trajectoryAgrees'])
        self.assertEqual([r['count'] for r in result['rows'][0]['decodeRows']], [4, 4])

    def test_f32_prefill_only_does_not_require_a_decode_protocol(self):
        output = self.root / 'prefill.json'
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.paths['gpu']), str(self.paths['reference']),
                                 str(output), '--kv-dtype', 'f32', '--inputs', str(self.paths['inputs']),
                                 '--policy', str(POLICY)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(output.read_bytes())
        self.assertTrue(receipt['passed'])
        self.assertEqual(receipt['rows'][0]['logits']['count'], 4)
        self.assertNotIn('decodeRows', receipt['rows'][0])

    def test_f32_decode_still_requires_an_explicit_protocol(self):
        output = self.root / 'missing-protocol.json'
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.paths['gpu']), str(self.paths['reference']),
                                 str(output), str(self.paths['decode']), str(self.paths['decode-inputs']),
                                 '--kv-dtype', 'f32', '--inputs', str(self.paths['inputs']), '--policy', str(POLICY)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires its explicit protocol', result.stderr)
        self.assertFalse(output.exists())

    def test_f32_prefill_rejects_missing_prompt_cases(self):
        self.gpu['rows'] = []
        self.save('gpu', self.gpu)
        output = self.root / 'empty-prefill.json'
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.paths['gpu']), str(self.paths['reference']),
                                 str(output), '--kv-dtype', 'f32', '--inputs', str(self.paths['inputs']),
                                 '--policy', str(POLICY)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing or reordered prefill cases', result.stderr)
        self.assertFalse(output.exists())

    def test_finite_wrong_decode_writes_a_failed_receipt(self):
        frame = self.gpu['rows'][0]['logitFrames'][1]
        values = self.values[1].copy()
        values[0] += 10
        Path(frame['file']).write_bytes(values.tobytes())
        frame['sha256'] = sha(Path(frame['file']).read_bytes())
        self.save('gpu', self.gpu)
        process, result = self.compare()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(result['passed'])
        self.assertGreater(result['rows'][0]['decodeRows'][0]['outsidePolicy'], 0)

    def test_unmatched_generated_prefix_is_not_compared_as_equal_input(self):
        self.gpu['rows'][0]['tokenId'] = 0
        self.gpu['rows'][0]['tokenIds'][0] = 0
        self.save('gpu', self.gpu)
        process, result = self.compare()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(result['passed'])
        self.assertTrue(all(not row['comparable'] for row in result['rows'][0]['decodeRows']))
        self.assertEqual(result['independentDecodeRowsChecked'], 0)

    def test_corrupted_source_payload_rejects_even_with_python_optimization(self):
        file = self.root / 'short-next.npy'
        raw = bytearray(file.read_bytes())
        raw[-1] ^= 1
        file.write_bytes(raw)
        for optimized in [False, True]:
            process, result = self.compare(optimized=optimized)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('Source tensor SHA-256 mismatch', process.stderr)
            self.assertIsNone(result)

    def test_missing_decode_step_rejects_before_receipt(self):
        del self.gpu['rows'][0]['logitFrames'][1]
        self.save('gpu', self.gpu)
        process, result = self.compare()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('logit steps', process.stderr)
        self.assertIsNone(result)

    def test_nonfinite_decode_cannot_pass_with_a_valid_hash(self):
        frame = self.gpu['rows'][0]['logitFrames'][1]
        values = self.values[1].copy()
        values[0] = np.nan
        Path(frame['file']).write_bytes(values.tobytes())
        frame['sha256'] = sha(Path(frame['file']).read_bytes())
        self.save('gpu', self.gpu)
        process, result = self.compare()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('Non-finite comparison', process.stderr)
        self.assertIsNone(result)

    def test_nearby_scores_do_not_establish_greedy_token_parity(self):
        source = np.array([0, 1, 1.001, 0], dtype='<f4')
        file = self.root / 'short.npy'
        np.save(file, source[None, :], allow_pickle=False)
        self.reference['sequences']['short']['logitsSha256'] = sha(file.read_bytes())
        self.save('reference', self.reference)
        self.gpu['referenceSha256'] = sha(self.paths['reference'].read_bytes())
        self.protocol['originalReferenceSha256'] = self.gpu['referenceSha256']
        self.save('protocol', self.protocol)
        actual = source.copy()
        actual[1] = 1.002
        frame = self.gpu['rows'][0]['logitFrames'][0]
        Path(frame['file']).write_bytes(actual.tobytes())
        frame['sha256'] = sha(Path(frame['file']).read_bytes())
        self.gpu['rows'][0]['prefillLogitsSha256'] = frame['sha256']
        self.save('gpu', self.gpu)
        process, result = self.compare()
        self.assertNotEqual(process.returncode, 0)
        self.assertTrue(result['rows'][0]['logits']['passed'])
        self.assertFalse(result['rows'][0]['passed'])
        self.assertEqual(result['rows'][0]['logits']['actualArgmax'], 1)

    def test_existing_evidence_is_not_overwritten(self):
        process, result = self.compare()
        self.assertEqual(process.returncode, 0, process.stderr)
        file = self.root / 'result-False.json'
        saved = file.read_bytes()
        process, _ = self.compare()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('FileExistsError', process.stderr)
        self.assertEqual(file.read_bytes(), saved)

    def test_counterfactual_and_wrong_lane_cannot_qualify(self):
        self.gpu['routingControl'] = dict(completed=True)
        self.save('gpu', self.gpu)
        process, result = self.compare()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('Substituted execution', process.stderr)
        self.assertIsNone(result)
        del self.gpu['routingControl']
        self.save('gpu', self.gpu)
        self.reference['kvStorageDtype'] = 'f16'
        self.save('reference', self.reference)
        self.gpu['referenceSha256'] = sha(self.paths['reference'].read_bytes())
        self.save('gpu', self.gpu)
        process, result = self.compare()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('KV lane mismatch', process.stderr)
        self.assertIsNone(result)

    def test_legacy_f16_consumer_frame_contract_is_preserved(self):
        del self.reference['inputsSha256']
        for name, value in [('reference', self.reference), ('decode', self.decode)]:
            value['kvStorageDtype'] = 'f16'
            self.save(name, value)
        row = self.gpu['rows'][0]
        row['kvStorageDtype'] = 'f16'
        row['stepLogits'] = [dict(f, inputTokenCount=2 + f['step'], tokenId=row['tokenIds'][f['step']]) for f in row.pop('logitFrames')]
        self.gpu['referenceSha256'] = sha(self.paths['reference'].read_bytes())
        self.save('gpu', self.gpu)
        process, result = self.compare(legacy=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(result['passed'])
        self.assertEqual(result['kvStorageDtype'], 'f16')


if __name__ == '__main__':
    unittest.main()
