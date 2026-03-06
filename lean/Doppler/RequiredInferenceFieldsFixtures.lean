import Doppler.RequiredInferenceFields

def requiredInferenceFieldsChecks : List (String × Bool) := [
  ("required_fields_nonnullable_missing_rejected",
    satisfiesNonNullableRequired .missing == false),
  ("required_fields_nonnullable_null_rejected",
    satisfiesNonNullableRequired .nullish == false),
  ("required_fields_nullable_missing_rejected",
    satisfiesNullableRequired .missing == false),
  ("required_fields_nullable_null_accepted",
    satisfiesNullableRequired .nullish == true)
]
