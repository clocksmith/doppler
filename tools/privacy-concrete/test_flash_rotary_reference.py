"""Pinned default text rotary positions used by the actual QSA component."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in pinned source/Torch reference')
class RotaryReferenceTests(unittest.TestCase):
    def test_default_text_positions_and_decode_prefix_match_explicit_frequencies(self):
        import torch
        from flash_rotary import text_rotary
        path=Path(os.environ['DOPPLER_FLASH_SOURCE_DIR'])/'modeling_qwen4_exp.py'
        config=SimpleNamespace(max_position_embeddings=8,head_dim=256,rope_parameters=dict(rope_type='default',rope_theta=10000000,partial_rotary_factor=.25,mrope_section=[11,11,10]))
        cos,sin=text_rotary(path,config,7)
        angles=torch.arange(7,dtype=torch.float32).reshape(1,-1,1)/(10000000**(torch.arange(0,64,2,dtype=torch.float32)/64))
        torch.testing.assert_close(cos,torch.cat([angles.cos(),angles.cos()],dim=-1),rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(sin,torch.cat([angles.sin(),angles.sin()],dim=-1),rtol=1e-6,atol=1e-6)
        earlier=text_rotary(path,config,4)
        torch.testing.assert_close(cos[:,:4],earlier[0],rtol=0,atol=0)
        with self.assertRaises(ValueError):text_rotary(path,config,9)
        config.rope_parameters['rope_type']='unsupported'
        with self.assertRaises(ValueError):text_rotary(path,config,1)


if __name__=='__main__':unittest.main()
