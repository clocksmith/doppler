inductive MergeValue where
  | missing
  | nullVal
  | enabled
  | disabled
  deriving Repr, DecidableEq

def nullishChoice : MergeValue → MergeValue → MergeValue
  | base, .missing => base
  | base, .nullVal => base
  | _, override => override

def definedOverlay : MergeValue → MergeValue → MergeValue
  | base, .missing => base
  | _, override => override

def spreadField : MergeValue → MergeValue → MergeValue
  | base, .missing => base
  | _, override => override

def subtreeReplace : MergeValue → MergeValue → MergeValue
  | base, .missing => base
  | base, .nullVal => base
  | _, override => override

theorem nullish_null_falls_through (base : MergeValue) :
    nullishChoice base .nullVal = base := by
  cases base <;> rfl

theorem nullish_missing_falls_through (base : MergeValue) :
    nullishChoice base .missing = base := by
  cases base <;> rfl

theorem defined_overlay_preserves_null (base : MergeValue) :
    definedOverlay base .nullVal = .nullVal := by
  cases base <;> rfl

theorem defined_overlay_missing_falls_through (base : MergeValue) :
    definedOverlay base .missing = base := by
  cases base <;> rfl

theorem spread_preserves_null (base : MergeValue) :
    spreadField base .nullVal = .nullVal := by
  cases base <;> rfl

theorem subtree_replace_uses_override (base override : MergeValue) :
    override ≠ .missing →
    override ≠ .nullVal →
    subtreeReplace base override = override := by
  intro hmissing hnull
  cases override <;> cases base <;> simp [subtreeReplace] at hmissing hnull ⊢

theorem subtree_replace_null_falls_through (base : MergeValue) :
    subtreeReplace base .nullVal = base := by
  cases base <;> rfl
