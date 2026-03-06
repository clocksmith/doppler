import Doppler.Quantization

def quantizationChecks : List (String × Bool) := [
  ("quantization_pad_ge_input", decide (padToQ4KBlock 257 >= 257)),
  ("quantization_pad_aligns_to_qk_k", padToQ4KBlock 257 % QK_K == 0),
  ("quantization_pad_is_idempotent", padToQ4KBlock (padToQ4KBlock 300) == padToQ4KBlock 300),
  ("quantization_block_count_covers_input", decide (q4kBlockCount 257 * QK_K >= 257))
]
