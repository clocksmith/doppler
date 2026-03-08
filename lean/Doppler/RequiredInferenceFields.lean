inductive FieldState where
  | missing
  | nullish
  | present
  deriving Repr, DecidableEq

inductive NullableValidatedFieldState where
  | missing
  | nullish
  | valid
  | invalid
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

def satisfiesNullablePositiveRequired : NullableValidatedFieldState → Bool
  | .nullish => true
  | .valid => true
  | _ => false

def satisfiesNullableUnitIntervalRequired : NullableValidatedFieldState → Bool
  | .nullish => true
  | .valid => true
  | _ => false

theorem nullable_positive_missing_rejected :
    satisfiesNullablePositiveRequired .missing = false := by
  decide

theorem nullable_positive_null_accepted :
    satisfiesNullablePositiveRequired .nullish = true := by
  decide

theorem nullable_positive_invalid_rejected :
    satisfiesNullablePositiveRequired .invalid = false := by
  decide

theorem nullable_positive_valid_accepted :
    satisfiesNullablePositiveRequired .valid = true := by
  decide

theorem nullable_unit_interval_missing_rejected :
    satisfiesNullableUnitIntervalRequired .missing = false := by
  decide

theorem nullable_unit_interval_null_accepted :
    satisfiesNullableUnitIntervalRequired .nullish = true := by
  decide

theorem nullable_unit_interval_invalid_rejected :
    satisfiesNullableUnitIntervalRequired .invalid = false := by
  decide

theorem nullable_unit_interval_valid_accepted :
    satisfiesNullableUnitIntervalRequired .valid = true := by
  decide
