inductive FieldState where
  | missing
  | nullish
  | present
  deriving Repr, DecidableEq

def satisfiesNonNullableRequired : FieldState → Bool
  | .present => true
  | _ => false

def satisfiesNullableRequired : FieldState → Bool
  | .missing => false
  | _ => true

theorem nonnullable_missing_rejected :
    satisfiesNonNullableRequired .missing = false := by
  decide

theorem nonnullable_null_rejected :
    satisfiesNonNullableRequired .nullish = false := by
  decide

theorem nullable_missing_rejected :
    satisfiesNullableRequired .missing = false := by
  decide

theorem nullable_null_accepted :
    satisfiesNullableRequired .nullish = true := by
  decide
