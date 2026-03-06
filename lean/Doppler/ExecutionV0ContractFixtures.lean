import Doppler.ExecutionV0Contract

def executionV0ContractChecks : List (String × Bool) := [
  ("executionv0_step_precision_precedes_profile",
    resolvePrecisionField (some .f16) (some .f32) .f32 = .f16),
  ("executionv0_profile_precision_precedes_session",
    resolvePrecisionField none (some .f16) .f32 = .f16),
  ("executionv0_session_precision_used_when_no_overrides",
    resolvePrecisionField none none .f32 = .f32),
  ("executionv0_pinned_profile_exact_once", exactPinnedProfileCount 1 = true),
  ("executionv0_pinned_profile_rejects_missing", exactPinnedProfileCount 0 = false),
  ("executionv0_pinned_profile_rejects_duplicates", exactPinnedProfileCount 2 = false)
]
