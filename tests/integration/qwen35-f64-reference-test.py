#!/usr/bin/env python3
"""Synthetic equation controls; these do not qualify physical inference."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('f64_reference', Path(__file__).parents[2] / 'tools/qwen35-f64-reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class Equations(unittest.TestCase):
    def test_delta_history_matches_independent_source(self):
        import torch
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, Qwen3_5TextConfig
        torch.set_num_threads(1)
        torch.manual_seed(901)
        config = Qwen3_5TextConfig(hidden_size=4, intermediate_size=8, num_attention_heads=2,
                                  num_key_value_heads=1, head_dim=2, num_hidden_layers=1,
                                  layer_types=['linear_attention'], linear_num_key_heads=1,
                                  linear_num_value_heads=2, linear_key_head_dim=2,
                                  linear_value_head_dim=3, linear_conv_kernel_dim=3)
        layer = Qwen3_5GatedDeltaNet(config, 0).eval()
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.2)
            x = torch.randn(1, 7, 4)
            expected = layer(x).numpy()[0]
        array = lambda tensor: tensor.detach().numpy().astype(np.float64)
        value = array(x[0])
        core = reference.delta_history(value @ array(layer.in_proj_qkv.weight).T,
                                       value @ array(layer.in_proj_z.weight).T,
                                       value @ array(layer.in_proj_a.weight).T,
                                       value @ array(layer.in_proj_b.weight).T,
                                       array(layer.conv1d.weight).squeeze(1), array(layer.A_log),
                                       array(layer.dt_bias), array(layer.norm.weight), config.to_dict())
        actual = core @ array(layer.out_proj.weight).T
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_cache_ties_are_even(self):
        first, second, third = np.array([1, 1 + 2 ** -10, 1 + 2 ** -9], dtype=np.float64)
        np.testing.assert_array_equal(reference.half_cache(np.array([(first + second) / 2, (second + third) / 2])), [first, third])

    def test_cache_rejects_overflow(self):
        with np.errstate(over='ignore'), self.assertRaisesRegex(AssertionError, 'nonfinite'):
            reference.half_cache(np.array([1e10]))

    def test_causal_attention_does_not_read_future(self):
        query = np.zeros((3, 2, 2))
        key = np.ones((3, 1, 2))
        value = np.array([[[2., 4.]], [[6., 8.]], [[40., 60.]]])
        actual = reference.causal_attention(query, key, value)
        np.testing.assert_allclose(actual[:, 0], [[2, 4], [4, 6], [16, 24]])
        np.testing.assert_array_equal(actual[:, 0], actual[:, 1])
        np.testing.assert_array_equal(actual[:2], reference.causal_attention(query[:2], key[:2], value[:2]))

    def test_rope_rotation_and_unrotated_tail(self):
        value = np.array([[[1., 0., 7.]], [[1., 0., 9.]]])
        actual = reference.text_rope(value, 10000, 2)
        np.testing.assert_allclose(actual[1, 0], [np.cos(1), np.sin(1), 9])
        np.testing.assert_array_equal(actual[0], value[0])

    def test_norm_offset_is_explicit(self):
        value, weight = np.array([[3., 4.]]), np.array([2., 3.])
        expected = value / np.sqrt(12.5 + 1e-6)
        np.testing.assert_allclose(reference.rms_norm(value, weight, 1e-6), expected * (weight + 1))
        np.testing.assert_allclose(reference.rms_norm(value, weight, 1e-6, False), expected * weight)

    def test_recurrence_has_zero_initial_state_and_prefix_causality(self):
        config = {'linear_num_key_heads': 1, 'linear_num_value_heads': 1,
                  'linear_key_head_dim': 1, 'linear_value_head_dim': 1, 'rms_norm_eps': 1e-6}
        qkv = np.array([[1., 2., 3.], [4., 5., 6.]])
        z, a, b = np.ones((2, 1)), np.zeros((2, 1)), np.zeros((2, 1))
        conv, log, bias, norm = np.ones((3, 1)), np.zeros(1), np.zeros(1), np.ones(1)
        actual = reference.delta_history(qkv, z, a, b, conv, log, bias, norm, config)
        np.testing.assert_array_equal(actual[:1], reference.delta_history(qkv[:1], z[:1], a[:1], b[:1], conv, log, bias, norm, config))
        q, k, v = qkv[0] / (1 + np.exp(-qkv[0]))
        q, k = q / np.sqrt(q*q + 1e-6), k / np.sqrt(k*k + 1e-6)
        core = q * k * v * 0.5
        expected = core / np.sqrt(core*core + 1e-6) / (1 + np.exp(-1))
        self.assertAlmostEqual(actual[0, 0], expected, places=14)

    def test_complete_hybrid_model_is_causal(self):
        config = {'hidden_act': 'silu', 'attention_bias': False, 'attn_output_gate': True,
                  'attention_dropout': 0, 'rms_norm_eps': 1e-6,
                  'layer_types': ['linear_attention', 'full_attention'],
                  'linear_num_key_heads': 1, 'linear_num_value_heads': 1,
                  'linear_key_head_dim': 2, 'linear_value_head_dim': 2,
                  'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 2,
                  'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000, 'partial_rotary_factor': 1}}
        rng = np.random.default_rng(741)
        make = lambda shape: rng.normal(0, 0.2, shape).astype(np.float64)
        weights = {'embed_tokens.weight': make((5, 4)), 'lm_head.weight': make((5, 4)), 'norm.weight': make(4)}
        for index in range(2):
            shapes = {'input_layernorm.weight': (4,), 'post_attention_layernorm.weight': (4,),
                      'mlp.gate_proj.weight': (6, 4), 'mlp.up_proj.weight': (6, 4), 'mlp.down_proj.weight': (4, 6)}
            if index == 0:
                shapes.update({'linear_attn.in_proj_qkv.weight': (6, 4), 'linear_attn.in_proj_z.weight': (2, 4),
                               'linear_attn.in_proj_a.weight': (1, 4), 'linear_attn.in_proj_b.weight': (1, 4),
                               'linear_attn.conv1d.weight': (6, 1, 2), 'linear_attn.A_log': (1,),
                               'linear_attn.dt_bias': (1,), 'linear_attn.norm.weight': (2,),
                               'linear_attn.out_proj.weight': (4, 2)})
            else:
                shapes.update({'self_attn.q_proj.weight': (8, 4), 'self_attn.k_proj.weight': (2, 4),
                               'self_attn.v_proj.weight': (2, 4), 'self_attn.o_proj.weight': (4, 4),
                               'self_attn.q_norm.weight': (2,), 'self_attn.k_norm.weight': (2,)})
            weights.update({f'layers.{index}.{key}': make(shape) for key, shape in shapes.items()})
        full, _ = reference.evaluate([1, 3, 2], weights, config, [0, 1, 2])
        for length in (1, 2):
            prefix, _ = reference.evaluate([1, 3, 2][:length], weights, config, [length - 1])
            np.testing.assert_allclose(prefix[0], full[length - 1], rtol=0, atol=1e-14)
        changed = dict(weights)
        changed['lm_head.weight'] = weights['lm_head.weight'] * 2
        doubled, _ = reference.evaluate([1, 3, 2], changed, config, [0, 1, 2])
        np.testing.assert_array_equal(doubled, full * 2)


if __name__ == '__main__':
    unittest.main()
