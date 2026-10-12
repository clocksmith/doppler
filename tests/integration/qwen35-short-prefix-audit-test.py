#!/usr/bin/env python3
"""Capture pairing checks; no model or physical qualification."""
import base64
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('short_audit',
    Path(__file__).parents[2] / 'tools/qwen35-short-prefix-audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(name, count, value):
    data = np.full((count, 2), value, dtype='<f4')
    return {'opId': name, 'capture': {'level': 'full', 'encoding': 'base64-f32le',
        'shape': list(data.shape), 'dataBase64': base64.b64encode(data.tobytes()).decode()}}


class Pairing(unittest.TestCase):
    def paired_item(self):
        pairs = np.array([[1., 2. ** -30], [0., -0.]], dtype='<f4')
        raw = pairs.tobytes()
        logical = pairs.astype(np.float64).sum(axis=1).astype('<f4')
        return {'encoding': 'base64-f32le', 'dtype': 'f32', 'precision': 'two-f32',
            'components': 2, 'elementStrideBytes': 8, 'shape': [1, 2],
            'storageEncoding': 'base64-f32le', 'storageDataBase64': base64.b64encode(raw).decode(),
            'storageDigest': 'sha256:' + hashlib.sha256(raw).hexdigest(),
            'dataBase64': base64.b64encode(logical.tobytes()).decode()}

    def test_paired_capture_preserves_residual_and_validates_raw_identity(self):
        item = self.paired_item()
        result = audit.decode_capture_values(item)
        self.assertEqual(result[0, 0], 1. + 2. ** -30)
        self.assertNotEqual(result[0, 0], np.frombuffer(base64.b64decode(item['dataBase64']), dtype='<f4')[0])
        for field, value in [('storageDigest', 'sha256:wrong'), ('components', 1),
                             ('elementStrideBytes', 4), ('dtype', 'f16'), ('precision', 'f32'),
                             ('storageDataBase64', base64.b64encode(bytes(8)).decode()),
                             ('dataBase64', base64.b64encode(np.ones(2, dtype='<f4').tobytes()).decode())]:
            with self.subTest(field=field), self.assertRaises(AssertionError):
                audit.decode_capture_values({**item, field: value})

    def test_paired_legacy_array_requires_exact_digest_including_signed_zero(self):
        item = self.paired_item()
        item['storageComponents'] = np.frombuffer(base64.b64decode(item.pop('storageDataBase64')), dtype='<f4').tolist()
        audit.decode_capture_values(item)
        item['storageComponents'][-1] = 0.
        with self.assertRaisesRegex(AssertionError, 'storage digest'):
            audit.decode_capture_values(item)
        del item['storageComponents']
        with self.assertRaises(KeyError):
            audit.decode_capture_values(item)

    def test_paired_nonfinite_storage_rejects_even_with_matching_digest(self):
        item = self.paired_item()
        raw = np.array([1., np.inf, 0., 0.], dtype='<f4').tobytes()
        item.update(storageDataBase64=base64.b64encode(raw).decode(),
                    storageDigest='sha256:' + hashlib.sha256(raw).hexdigest())
        with self.assertRaises(AssertionError):
            audit.decode_capture_values(item)

    def test_paired_duplicate_cannot_hide_different_tiny_residuals(self):
        item = self.paired_item()
        raw = np.array([[1., 0.], [0., 0.]], dtype='<f4')
        def encoded(value):
            return {**item, 'level': 'full', 'dataBase64': base64.b64encode(np.array([1., 0.], dtype='<f4').tobytes()).decode(),
                'storageDataBase64': base64.b64encode(value.tobytes()).decode(),
                'storageDigest': 'sha256:' + hashlib.sha256(value.tobytes()).hexdigest()}
        first = encoded(raw)
        raw[0, 1] = np.nextafter(np.float32(0), np.float32(1))
        second = encoded(raw)
        np.testing.assert_array_equal(audit.decode_capture_values(first), audit.decode_capture_values(second))
        capture = {'observation': {'timeline': [{'opId': 'embed.out', 'capture': first},
            {'opId': 'layer.0.layer.out', 'capture': first}, {'opId': 'layer.0.layer.out', 'capture': second}]}}
        with self.assertRaises(AssertionError):
            audit.read_capture(capture, prefill=1, decode_steps=0)

    def test_detailed_layer_selection_is_explicit_and_validated(self):
        self.assertEqual(audit.selected_audit_layers(None, 24), list(range(24)))
        self.assertEqual(audit.selected_audit_layers([0], 24), [0])
        self.assertEqual(audit.selected_audit_layers([3, 0], 24), [0, 3])
        for invalid in [[], [24], [-1], [True], [0, 0], [0.5]]:
            with self.subTest(layers=invalid), self.assertRaises(AssertionError):
                audit.selected_audit_layers(invalid, 24)

    def test_scalar_intervention_is_explicit_and_separate_from_tensor_stores(self):
        epsilon = 1e-6
        selection = audit.StoreIntervention([], 24, 3, [2, 3, 4], ['rms-epsilon'])
        self.assertEqual(selection.uniform(epsilon, 'rms-epsilon'), epsilon)
        self.assertNotEqual(selection.uniform(epsilon, 'recurrent-rms-epsilon'), epsilon)
        self.assertEqual(selection.uniform(epsilon, 'recurrent-l2-epsilon'), float(np.float32(epsilon)))
        np.testing.assert_array_equal(selection.round(np.array([1 + 2 ** -30]), 'normalization'), [1.])
        self.assertEqual(selection.receipt()['uniformCalls'], {'rms-epsilon': 1})
        with self.assertRaises(AssertionError):
            audit.StoreIntervention([], 24, 3, [2], ['unknown'])
        with self.assertRaises(AssertionError):
            audit.StoreIntervention([], 24, 3, [2], ['rms-epsilon']).receipt()

    def test_intervention_is_scoped_to_layer_operation_and_token_phase(self):
        value = np.full((5, 2), 1.0 + 2 ** -30)
        rounded = value.astype(np.float32).astype(np.float64)
        selection = audit.StoreIntervention([{'operations': ['q-projection'],
            'layers': [3], 'phases': ['decode']}], 24, 3, [2, 3, 4])
        selection.set_context(layer=2)
        np.testing.assert_array_equal(selection.round(value, 'q-projection'), rounded)
        selection.set_context(layer=3)
        np.testing.assert_array_equal(selection.round(value, 'k-projection'), rounded)
        expected = rounded.copy()
        expected[3:] = value[3:]
        np.testing.assert_array_equal(selection.round(value, 'q-projection'), expected)
        self.assertEqual(selection.receipt()['selectedElements'], [4])

    def test_intervention_handles_output_positions_and_scalar_recurrence(self):
        value = np.full((3, 2), 1.0 + 2 ** -30)
        selection = audit.StoreIntervention([{'operations': None,
            'layers': None, 'phases': ['decode']}], 24, 3, [2, 3, 4])
        selection.set_context(layer='output')
        expected = value.copy()
        expected[0] = 1.0
        np.testing.assert_array_equal(selection.round(value, 'logits'), expected)
        selection.set_context(layer=0, token=2)
        np.testing.assert_array_equal(selection.round(value, 'gated-recurrent-output'), np.ones_like(value))
        selection.set_context(token=3)
        np.testing.assert_array_equal(selection.round(value, 'gated-recurrent-output'), value)
        np.testing.assert_array_equal(selection.round(value, 'recurrent-high'), np.ones_like(value))
        np.testing.assert_array_equal(selection.round(value, 'recurrent-low'), np.ones_like(value))

    def test_intervention_rejects_mistyped_or_unobserved_selectors(self):
        row = {'operations': ['k-projection'], 'layers': [3], 'phases': ['prefill']}
        for key, invalid in [('operations', ['unknown']), ('operations', ['recurrent-high']),
                             ('layers', [24]), ('layers', [True]), ('phases', []),
                             ('phases', ['decode', 'decode']), ('phases', ['prompt'])]:
            with self.assertRaises(AssertionError):
                audit.StoreIntervention([{**row, key: invalid}], 24, 3, [2, 3, 4])
        selection = audit.StoreIntervention([row], 24, 3, [2, 3, 4])
        with self.assertRaises(AssertionError):
            selection.receipt()

    def test_empty_intervention_preserves_stores_and_overlaps_count_union(self):
        value = np.full((5, 2), 1.0 + 2 ** -30)
        selection = audit.StoreIntervention([], 24, 3, [2, 3, 4])
        np.testing.assert_array_equal(selection.round(value, 'embedding'), np.ones_like(value))
        self.assertEqual(selection.receipt()['selectedElements'], [])
        row = {'operations': ['embedding'], 'layers': None, 'phases': None}
        selection = audit.StoreIntervention([row, {**row, 'phases': ['decode']}], 24, 3, [2, 3, 4])
        np.testing.assert_array_equal(selection.round(value, 'embedding'), value)
        self.assertEqual(selection.receipt()['selectedElements'], [10, 4])
        self.assertEqual(selection.receipt()['suppressedStores'], {'input:embedding': 10})

    def test_ideal_materialization_rejects_wrong_archive_and_capture(self):
        digest = lambda value: hashlib.sha256(value).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            dispatch_path = Path(directory) / 'dispatch.json'
            dispatch_path.write_text(json.dumps({'archiveSha256': 'archive', 'captureSha256': 'capture'}))
            policy = {'reviewStatus': 'reviewed', 'evaluatorSha256': audit.EVALUATOR_SHA256,
                'parentArchiveSha256': 'archive', 'dispatchBinding': {
                    'path': str(dispatch_path), 'sha256': digest(dispatch_path.read_bytes())}}
            map_path = Path(directory) / 'map.json'
            map_path.write_text(json.dumps(policy))
            for archive, capture in [('wrong', 'capture'), ('archive', 'wrong')]:
                with self.assertRaises(AssertionError):
                    audit.ideal_global_materialization(SimpleNamespace(digest=digest),
                        None, None, None, None, map_path, archive, capture)
            dispatch_path.write_text('{}')
            with self.assertRaises(AssertionError):
                audit.ideal_global_materialization(SimpleNamespace(digest=digest),
                    None, None, None, None, map_path, 'archive', 'capture')

    def test_materialization_history_cannot_exceed_frozen_contract(self):
        digest = lambda value: hashlib.sha256(value).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            dispatch_path = Path(directory) / 'dispatch.json'
            dispatch_path.write_text(json.dumps({'archiveSha256': 'archive', 'captureSha256': 'capture'}))
            policy = {'reviewStatus': 'reviewed', 'evaluatorSha256': audit.EVALUATOR_SHA256,
                'parentArchiveSha256': 'archive', 'dispatchBinding': {
                    'path': str(dispatch_path), 'sha256': digest(dispatch_path.read_bytes())},
                'request': {'index': 2, 'prefillTokens': 184, 'lastDecodeStep': 7}}
            map_path = Path(directory) / 'map.json'
            map_path.write_text(json.dumps(policy))
            contract = {'requestIndex': 2, 'prefixLengths': list(range(184, 235))}
            for count in [0, -1, 52, True, 8.5]:
                with self.assertRaises(AssertionError):
                    audit.ideal_global_materialization(SimpleNamespace(digest=digest),
                        None, None, contract, None, map_path, 'archive', 'capture', prefix_count=count)

    def test_runtime_comparison_rejects_different_forced_history(self):
        binding = {'archiveSha256': 'archive', 'manifestSha256': 'manifest'}
        contract = {'requestIndex': 0, 'steps': [0, 1], 'prefixLengths': [2, 3], 'inputIds': [10, 11, 12]}
        history = {'modelIdentity': 'frozen', 'expected': [{'steps': [{'tokenId': 12}, {'tokenId': 13}]}]}
        runtime = {'archiveSha256': 'archive', 'manifestIdentity': 'sha256:manifest',
            'referenceSha256': 'history', 'frozenManifestIdentity': 'frozen', 'fullPrefixes': True,
            'physicalExecutionCompleted': True, 'declaredShaderMismatches': [],
            'results': [{'index': 0, 'step': step, 'inputIds': [10, 11]} for step in [0, 1]]}
        self.assertEqual(len(audit.validate_runtime_context(runtime, binding, contract, history, 'history')), 2)
        for key, value in [('referenceSha256', 'other'), ('manifestIdentity', 'other'), ('fullPrefixes', False)]:
            wrong = {**runtime, key: value}
            with self.assertRaises(AssertionError):
                audit.validate_runtime_context(wrong, binding, contract, history, 'history')
        wrong = copy.deepcopy(history)
        wrong['expected'][0]['steps'][0]['tokenId'] = 99
        with self.assertRaises(AssertionError):
            audit.validate_runtime_context(runtime, binding, contract, wrong, 'history')

    def test_equal_maxima_do_not_imply_same_error_source(self):
        expected = np.zeros((1, 4))
        actual = np.array([[.002, -.002, 0., 0.]])
        opposite = -actual
        result = audit.compare_error_sources(actual, opposite, expected)
        self.assertEqual(result['tailOverlap'], 2)
        self.assertEqual(result['sameSignTailOverlap'], 0)
        self.assertAlmostEqual(result['signedErrorCorrelation'], -1.)
        self.assertAlmostEqual(result['actualAgainstControlled']['max'], .004)
        disjoint = np.array([[0., 0., .002, -.002]])
        result = audit.compare_error_sources(actual, disjoint, expected)
        self.assertEqual(result['tailOverlap'], 0)
        self.assertAlmostEqual(result['signedErrorCosine'], 0.)

    def test_ideal_f32_can_double_round_before_half_store(self):
        midpoint = 1.0 + 2.0 ** -11
        values = np.array([[midpoint + 2.0 ** -26, midpoint - 2.0 ** -26, 1.0]], dtype=np.float64)
        result = audit.cache_double_rounding(values)
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['coordinates'][0]['index'], [0, 0])
        self.assertEqual(result['coordinates'][0]['nearestF32'], midpoint)
        self.assertEqual(result['coordinates'][0]['throughF32F16'], 1.0)
        self.assertEqual(result['coordinates'][0]['directF16'], 1.0 + 2.0 ** -10)

    def capture(self):
        return {'observation': {'timeline': [row('embed.out', 19, 1),
            row('layer.0.layer.out', 19, 2), row('layer.0.layer.out', 19, 2),
            row('layer.0.ffn.gate', 19, 3), row('embed.out', 1, 4),
            row('layer.0.layer.out', 1, 5)]}}

    def test_pair_by_step_and_preserve_decode(self):
        result = audit.read_capture(self.capture())
        np.testing.assert_array_equal(result['layer.0.layer.out'][:, 0], [2] * 19 + [5])
        self.assertNotIn('layer.0.ffn.gate', result)

    def test_longer_prefix_pairs_every_decode_step(self):
        timeline = []
        for step, count in enumerate([184, 1, 1, 1]):
            timeline.extend([row('embed.out', count, step),
                             row('layer.0.layer.out', count, step + 10)])
        result = audit.read_capture({'observation': {'timeline': timeline}}, 184, 3)
        np.testing.assert_array_equal(result['layer.0.layer.out'][:, 0],
                                      [10] * 184 + [11, 12, 13])
        del timeline[5]
        with self.assertRaises(AssertionError):
            audit.read_capture({'observation': {'timeline': timeline}}, 184, 3)

    def test_conflicting_duplicate_rejected(self):
        capture = self.capture()
        capture['observation']['timeline'][2] = row('layer.0.layer.out', 19, 9)
        with self.assertRaises(AssertionError):
            audit.read_capture(capture)

    def test_missing_decode_not_silently_replaced_with_prefill(self):
        capture = self.capture()
        capture['observation']['timeline'].pop()
        with self.assertRaises(AssertionError):
            audit.read_capture(capture)

    def test_token_count_mismatch_rejected(self):
        capture = self.capture()
        capture['observation']['timeline'][-1] = row('layer.0.layer.out', 2, 5)
        with self.assertRaises(AssertionError):
            audit.read_capture(capture)

    def test_distribution_keeps_tail_and_nonfinite_failure(self):
        result = audit.error(np.array([[0., .0001], [.0009, .002]]), np.zeros((2, 2)))
        self.assertEqual(result['overTolerance'], 1)
        self.assertEqual(result['lastTokenMax'], .002)
        with self.assertRaises(AssertionError):
            audit.error(np.array([np.nan]), np.zeros(1))


if __name__ == '__main__':
    unittest.main()
