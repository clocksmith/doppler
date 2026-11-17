"""Actual public quantizer metadata; no ciphertexts or native encryption."""
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in actual public quantizer')
class ProjectionIdentityTests(unittest.TestCase):
    def test_exact_fingerprint_rejects_value_shape_dtype_bias_and_nonfinite_changes(self):
        import torch
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        from full_model import FullLinear
        from projection_identity import ProjectionIdentity
        source=torch.nn.Linear(4,3,bias=True)
        with tempfile.TemporaryDirectory() as root:
            projection=FullLinear('public',source,GLWELinearLayerExecutor(use_dynamic_quantization=True),Path(root),12)
            projection.release_float_reference();identity=ProjectionIdentity.from_projection(projection);identity.require_projection(projection)
            for name in ('scale','zero_point','sum_weights','prepared_bias'):
                original=getattr(projection,name)
                for change in ('value','shape','dtype','nan','meta-device','missing'):
                    with self.subTest(name=name,change=change):
                        value=(original+1 if change=='value' else original.reshape(1,1,-1) if change=='shape'
                            else original.double() if change=='dtype' else torch.full_like(original,float('nan')) if change=='nan'
                            else torch.empty(original.shape,device='meta') if change=='meta-device' else None)
                        setattr(projection,name,value)
                        with self.assertRaises(ValueError):identity.require_projection(projection)
                        setattr(projection,name,original)
            identity.require_projection(projection)

    def test_nonfinite_or_missing_required_metadata_cannot_create_an_identity(self):
        import torch
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        from full_model import FullLinear
        from projection_identity import ProjectionIdentity
        with tempfile.TemporaryDirectory() as root:
            projection=FullLinear('public',torch.nn.Linear(4,3,bias=False),GLWELinearLayerExecutor(use_dynamic_quantization=True),Path(root),12)
            projection.release_float_reference();original=projection.scale
            for value in (None,torch.full_like(original,float('inf')),torch.full_like(original,float('nan'))):
                projection.scale=value
                with self.assertRaises(ValueError):ProjectionIdentity.from_projection(projection)


if __name__=='__main__':unittest.main()
