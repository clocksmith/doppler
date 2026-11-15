"""Pre-disclosure public-model admission checks; no model download or encrypted work."""
import os
import unittest
from unittest.mock import Mock, patch


@unittest.skipUnless(os.environ.get('DOPPLER_CONCRETE_NATIVE') == '1', 'opt-in pinned native dependencies')
class BridgeProfileTests(unittest.TestCase):
    def test_rectangular_geometry_and_weight_pin_reject_before_key_export_or_spawn(self):
        import numpy as np
        from bridge import EvaluatorBridge
        from profiles import load_profile
        from protocol import digest
        profile = load_profile('qwen05b-k-projection')
        for shape in [(16, 16), (896, 128)]:
            matrix = np.zeros(shape, dtype='<u8')
            key = Mock()
            with patch('bridge.subprocess.Popen') as process:
                with self.assertRaisesRegex(ValueError, 'Unsupported public projection'):
                    EvaluatorBridge('m' * 64, key, {digest(matrix.tobytes()): matrix}, profile=profile)
                key.serialize.assert_not_called()
                process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
