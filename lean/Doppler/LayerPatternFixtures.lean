import Doppler.LayerPattern

def layerPatternChecks : List (String × Bool) := [
  ("layer_pattern_alternating_even_even_layers_full",
    layerTypeForPattern .alternating_even true false == .full_attention),
  ("layer_pattern_alternating_even_odd_layers_sliding",
    layerTypeForPattern .alternating_even false false == .sliding_attention),
  ("layer_pattern_every_n_stride_layers_full",
    layerTypeForPattern .every_n false true == .full_attention),
  ("layer_pattern_every_n_non_stride_layers_sliding",
    layerTypeForPattern .every_n true false == .sliding_attention),
  ("layer_pattern_offset_normalizes_negative",
    normalizeEveryNOffset (-1) 6 == 5)
]
