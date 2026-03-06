def QK_K : Nat := 256
def Q4K_BLOCK_BYTES : Nat := 144
def Q6K_BLOCK_BYTES : Nat := 210
def Q8_0_BLOCK_BYTES : Nat := 34
def Q8_0_BLOCK_SIZE : Nat := 32
def K_SCALE_SIZE : Nat := 12

def padToQ4KBlock (size : Nat) : Nat :=
  ((size + QK_K - 1) / QK_K) * QK_K

def q4kBlockCount (numElements : Nat) : Nat :=
  (numElements + QK_K - 1) / QK_K

theorem pad_to_q4k_block_257_ge :
    padToQ4KBlock 257 ≥ 257 := by
  native_decide

theorem pad_to_q4k_block_257_aligned :
    padToQ4KBlock 257 % QK_K = 0 := by
  native_decide

theorem pad_to_q4k_block_300_idempotent :
    padToQ4KBlock (padToQ4KBlock 300) = padToQ4KBlock 300 := by
  native_decide

theorem q4k_block_count_257_covers :
    q4kBlockCount 257 * QK_K ≥ 257 := by
  native_decide
