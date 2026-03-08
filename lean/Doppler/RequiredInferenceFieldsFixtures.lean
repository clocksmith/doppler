import Doppler.RequiredInferenceFields

def requiredInferenceFieldsChecks : List (String × Bool) := [
  ("required_fields_nonnullable_missing_rejected",
    satisfiesNonNullableRequired .missing == false),
  ("required_fields_nonnullable_null_rejected",
    satisfiesNonNullableRequired .nullish == false),
  ("required_fields_nullable_missing_rejected",
    satisfiesNullableRequired .missing == false),
  ("required_fields_nullable_null_accepted",
    satisfiesNullableRequired .nullish == true),
  ("required_fields_nullable_positive_invalid_rejected",
    satisfiesNullablePositiveRequired .invalid == false),
  ("required_fields_nullable_positive_valid_accepted",
    satisfiesNullablePositiveRequired .valid == true),
  ("required_fields_nullable_unit_interval_invalid_rejected",
    satisfiesNullableUnitIntervalRequired .invalid == false),
  ("required_fields_nullable_unit_interval_valid_accepted",
    satisfiesNullableUnitIntervalRequired .valid == true)
]
