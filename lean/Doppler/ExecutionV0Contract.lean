import Doppler.Model

def exactPinnedProfileCount (count : Nat) : Bool :=
  decide (count = 1)

def resolvePrecisionField : Option Dtype → Option Dtype → Dtype → Dtype
  | some stepDtype, _, _ => stepDtype
  | none, some profileDtype, _ => profileDtype
  | none, none, sessionDtype => sessionDtype

theorem step_precision_precedes_profile
    (step profile : Dtype)
    (session : Dtype) :
    resolvePrecisionField (some step) (some profile) session = step := by
  rfl

theorem profile_precision_precedes_session
    (profile session : Dtype) :
    resolvePrecisionField none (some profile) session = profile := by
  rfl

theorem session_precision_used_when_no_overrides
    (session : Dtype) :
    resolvePrecisionField none none session = session := by
  rfl

theorem pinned_profile_exact_once :
    exactPinnedProfileCount 1 = true := by
  simp [exactPinnedProfileCount]

theorem pinned_profile_rejects_missing :
    exactPinnedProfileCount 0 = false := by
  simp [exactPinnedProfileCount]

theorem pinned_profile_rejects_duplicates :
    exactPinnedProfileCount 2 = false := by
  simp [exactPinnedProfileCount]
