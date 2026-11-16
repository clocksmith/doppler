import unittest
from full_admission import FullRouteAdmission
from full_plan import tiles


class Plan:
    def definition(self):return dict(steps=4,contextBucket=8)
    def schedule(self):
        return tiles(dict(name='first',weightId='a',inputWidth=16,outputWidth=16))+tiles(dict(name='last',weightId='b',inputWidth=16,outputWidth=16))


class FullAdmissionTests(unittest.TestCase):
    def test_all_steps_use_exact_order_and_fixed_prefill_decode_rows(self):
        gate=FullRouteAdmission(Plan())
        for step in range(4):
            for name in ['first','last']:
                gate.admit_projection(name,16,16,8 if step==0 else 1)
                tile=dict(gate.next_tile())
                gate.admit_tile(tile,(16,16),gate.position+1)
                gate.verified()
        gate.complete()

    def test_wrong_projection_or_private_shape_is_terminal_before_encryption(self):
        for name,rows in [('last',8),('first',5),('first',True)]:
            gate=FullRouteAdmission(Plan())
            with self.assertRaises(ValueError):gate.admit_projection(name,16,16,rows)
            with self.assertRaises(ValueError):gate.next_tile()

    def test_tile_substitution_and_retry_are_terminal_before_serialization(self):
        gate=FullRouteAdmission(Plan())
        tile=dict(gate.next_tile())
        with self.assertRaises(ValueError):gate.admit_tile(dict(tile,id='other'),(16,16),1)
        with self.assertRaises(ValueError):gate.admit_tile(tile,(16,16),1)
        gate=FullRouteAdmission(Plan())
        gate.admit_projection('first',16,16,8)
        gate.admit_tile(dict(gate.next_tile()),(16,16),1)
        with self.assertRaises(ValueError):gate.admit_projection('first',16,16,8)

    def test_tile_cannot_bypass_projection_admission(self):
        gate=FullRouteAdmission(Plan())
        with self.assertRaises(ValueError):gate.admit_tile(dict(gate.next_tile()),(16,16),1)
        self.assertTrue(gate.failed)

    def test_incomplete_or_failed_routes_cannot_finalize(self):
        gate=FullRouteAdmission(Plan())
        with self.assertRaises(ValueError):gate.complete()
        with self.assertRaises(ValueError):gate.next_tile()


if __name__=='__main__':unittest.main()
