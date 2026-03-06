import Doppler.MergeSemantics

def mergeSemanticsChecks : List (String × Bool) := [
  ("merge_nullish_null_falls_through", nullishChoice .enabled .nullVal = .enabled),
  ("merge_nullish_missing_falls_through", nullishChoice .enabled .missing = .enabled),
  ("merge_overlay_null_overrides", definedOverlay .enabled .nullVal = .nullVal),
  ("merge_overlay_missing_falls_through", definedOverlay .enabled .missing = .enabled),
  ("merge_spread_null_overrides", spreadField .enabled .nullVal = .nullVal),
  ("merge_subtree_override_replaces_base", subtreeReplace .enabled .disabled = .disabled),
  ("merge_subtree_null_falls_through", subtreeReplace .enabled .nullVal = .enabled)
]
