inductive KVLayout where
  | contiguous
  | paged
  | tiered
  | bdpa
  | bdpa_paged
  deriving Repr, DecidableEq

inductive Phase where
  | prefill
  | decode
  | both
  deriving Repr, DecidableEq

inductive Dtype where
  | f16
  | f32
  deriving Repr, DecidableEq

inductive OpClass where
  | attention
  | embed
  | norm
  | projection
  | residual
  | sample
  | other
  deriving Repr, DecidableEq

inductive ColdQuantMode where
  | none
  | int8
  | int4
  deriving Repr, DecidableEq

def Dtype.bytesPerElement : Dtype → Nat
  | .f16 => 2
  | .f32 => 4

def Dtype.rank : Dtype → Nat
  | .f16 => 1
  | .f32 => 2

theorem f32_bytes_ge_f16_bytes :
    Dtype.bytesPerElement .f32 ≥ Dtype.bytesPerElement .f16 := by
  decide

theorem f32_rank_ge_f16_rank :
    Dtype.rank .f32 ≥ Dtype.rank .f16 := by
  decide
