"""Installed-profile binding and arithmetic limits run without native crypto/model dependencies."""
from dataclasses import FrozenInstanceError
import unittest
from profiles import load_profile, validate_arithmetic_capacity
from protocol import check


class ProfileTests(unittest.TestCase):
    def test_profiles_are_immutable_and_each_specification_is_an_independent_copy(self):
        profile = load_profile('qwen05b-k-projection')
        with self.assertRaises(FrozenInstanceError):
            profile.input_width = 2560
        copy = profile.specification()
        copy['projection']['inputWidth'] = 1
        self.assertEqual(profile.specification()['projection']['inputWidth'], 896)
        self.assertFalse(copy['productionBackendAvailable'])

    def test_a_response_cannot_downgrade_or_cross_model_profiles(self):
        tiny, qwen = load_profile(), load_profile('qwen05b-k-projection')
        arguments = ('model', 'session', 'key', 2, 'result', 'weight')
        check(qwen.frame(*arguments), qwen.frame(*arguments))
        with self.assertRaises(ValueError):
            check(tiny.frame(*arguments), qwen.frame(*arguments))
        self.assertNotEqual(tiny.policy_id, qwen.policy_id)
        self.assertEqual((qwen.input_width, qwen.output_width), (896, 128))

    def test_unsupported_profiles_cannot_select_a_file_or_expand_capabilities(self):
        for name in ['../policy', 'Qwen3.8-Flash-Next', 'attested', 'plaintext', 'qwen05b-k-projection/other']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_profile(name)

    def test_flash_width_exceeds_the_pinned_conservative_signed_dot_product_range(self):
        self.assertEqual(validate_arithmetic_capacity(896, 8), 29245440)
        for width, bits in [(2560, 8), (896, 16), (True, 8), (0, 8)]:
            with self.subTest(width=width, bits=bits), self.assertRaises(ValueError):
                validate_arithmetic_capacity(width, bits)


if __name__ == '__main__':
    unittest.main()
