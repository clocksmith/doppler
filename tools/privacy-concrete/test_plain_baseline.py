"""CPU-only baseline reporting regressions; no model files or provider required."""
import importlib.util
import unittest
from unittest.mock import patch

# The protocol smoke job deliberately has no native model dependencies. Keep
# discovery usable there, while exercising these tests in the pinned CPU venv.
BASELINE_DEPENDENCIES = ('numpy', 'torch', 'transformers', 'concrete', 'concrete_ml_extensions')
HAS_BASELINE_DEPENDENCIES = all(importlib.util.find_spec(name) is not None for name in BASELINE_DEPENDENCIES)
if HAS_BASELINE_DEPENDENCIES:
    import torch
    import shares_model


@unittest.skipUnless(HAS_BASELINE_DEPENDENCIES, 'requires the pinned privacy-concrete CPU dependencies')
class PlainBaselineTest(unittest.TestCase):
    def report(self, quant_tokens, float_tokens):
        class Tokenizer:
            def encode(self, prompt): return [5]
            def decode(self, tokens): return str(tokens)
        class Plan:
            identity = 'plan'
            def definition(self): return {'modelId': 'model'}
        calls = []
        def step(model, projections, cache, history, original_count, index, mode, bucket, pad):
            expected = float_tokens if mode == 'float-reference' else quant_tokens
            self.assertEqual(history, [5] + expected[:index])
            calls.append((mode, index))
            logits = torch.zeros(4)
            logits[expected[index]] = 2
            return logits, 10.0 if mode == 'float-reference' else 20.0
        with patch.object(shares_model, 'qwen_backend', return_value=({'steps': len(quant_tokens), 'contextBucket': 8}, None, None, None)), \
             patch.object(shares_model, 'prepare', return_value=(None, Tokenizer(), None, [], Plan())), \
             patch.object(shares_model, 'model_step', side_effect=step):
            report = shares_model.run_plain_baseline('unused', 'unused', 'prompt')
        self.assertEqual(len(calls), 2 * len(quant_tokens))
        self.assertEqual(report['floatReferenceExecutionWallMs'], 10 * len(quant_tokens))
        self.assertEqual(report['clearQuantizedExecutionWallMs'], 20 * len(quant_tokens))
        self.assertEqual(report['outputTokens'], quant_tokens)
        self.assertEqual(report['floatOutputTokens'], float_tokens)
        return report

    def test_matching_histories_keep_comparisons_and_standalone_timings(self):
        report = self.report([1, 2, 3], [1, 2, 3])
        self.assertEqual(report['referenceInputHistoriesMatch'], [True, True, True])
        self.assertEqual(report['quantizedVsFloatMaxLogitErrors'], [0, 0, 0])
        self.assertEqual(report['floatGreedyMatches'], [True, True, True])

    def test_divergence_invalidates_later_comparisons_even_if_tokens_reconverge(self):
        report = self.report([1, 2, 3], [0, 2, 3])
        self.assertEqual(report['quantizedVsFloatMaxLogitErrors'], [2, None, None])
        self.assertEqual(report['referenceInputHistoriesMatch'], [True, False, False])
        self.assertEqual(report['floatGreedyMatches'], [False, None, None])

    def test_late_divergence_keeps_the_same_input_step_valid(self):
        report = self.report([1, 2, 3], [1, 0, 3])
        self.assertEqual(report['quantizedVsFloatMaxLogitErrors'], [0, 2, None])
        self.assertEqual(report['referenceInputHistoriesMatch'], [True, True, False])
        self.assertEqual(report['floatGreedyMatches'], [True, False, None])


if __name__ == '__main__':
    unittest.main()
