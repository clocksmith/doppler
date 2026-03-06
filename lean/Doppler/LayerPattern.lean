inductive PatternKind where
  | alternating_even
  | alternating_odd
  | every_n
  deriving Repr, DecidableEq

inductive PatternLayerType where
  | full_attention
  | sliding_attention
  deriving Repr, DecidableEq

def layerTypeForPattern (kind : PatternKind) (isEven isStride : Bool) : PatternLayerType :=
  match kind with
  | .alternating_even =>
    if isEven then .full_attention else .sliding_attention
  | .alternating_odd =>
    if isEven then .sliding_attention else .full_attention
  | .every_n =>
    if isStride then .full_attention else .sliding_attention

def normalizeEveryNOffset (rawOffset : Int) (period : Nat) : Nat :=
  if _h : period = 0 then
    0
  else
    (((rawOffset % period) + period) % period).toNat

theorem normalize_every_n_offset_negative_one_6 :
    normalizeEveryNOffset (-1) 6 = 5 := by
  native_decide

theorem normalize_every_n_offset_positive_in_range_example :
    normalizeEveryNOffset 5 6 = 5 := by
  native_decide

theorem alternating_even_even_layers_are_full :
    layerTypeForPattern .alternating_even true false = .full_attention := by
  decide

theorem alternating_even_odd_layers_are_sliding :
    layerTypeForPattern .alternating_even false false = .sliding_attention := by
  decide

theorem alternating_odd_even_layers_are_sliding :
    layerTypeForPattern .alternating_odd true false = .sliding_attention := by
  decide

theorem alternating_odd_odd_layers_are_full :
    layerTypeForPattern .alternating_odd false false = .full_attention := by
  decide

theorem every_n_stride_layers_are_full :
    layerTypeForPattern .every_n false true = .full_attention := by
  decide

theorem every_n_non_stride_layers_are_sliding :
    layerTypeForPattern .every_n true false = .sliding_attention := by
  decide
