import Doppler.ExecutionContract

def translateGemmaSteps : List ExecutionStep := [
  {
    id := "embed_tokens",
    phase := .both,
    opClass := .embed,
  },
  {
    id := "prefill_attention",
    phase := .prefill,
    opClass := .attention,
  },
  {
    id := "decode_attention",
    phase := .decode,
    opClass := .attention,
  }
]

def translateGemmaSession : SessionConfig := {
  layout := .bdpa,
  disableCommandBatching := false,
  decodeBatchSize := 16,
  headDim := 256,
  kvLen := 1024,
  coldQuantMode := .none,
}

def translateGemmaFixedSession : SessionConfig := {
  layout := .contiguous,
  disableCommandBatching := false,
  decodeBatchSize := 16,
  headDim := 256,
  kvLen := 1024,
  coldQuantMode := .none,
}

def executionContractChecks : List (String × Bool) := [
  ("translategemma_conflicting_steps", allStepsCompatible translateGemmaSteps translateGemmaSession),
  ("translategemma_conflicting_session", sessionConsistent translateGemmaSession),
  ("translategemma_fixed_steps", allStepsCompatible translateGemmaSteps translateGemmaFixedSession),
  ("translategemma_fixed_session", sessionConsistent translateGemmaFixedSession)
]

theorem translateGemma_conflicting_steps_fail :
    allStepsCompatible translateGemmaSteps translateGemmaSession = false := by
  native_decide

theorem translateGemma_conflicting_session_fails :
    sessionConsistent translateGemmaSession = false := by
  native_decide

theorem translateGemma_fixed_steps_pass :
    allStepsCompatible translateGemmaSteps translateGemmaFixedSession = true := by
  native_decide

theorem translateGemma_prefill_attention_rules_out_bdpa_for_compatible_sessions :
    allStepsCompatible translateGemmaSteps translateGemmaSession = true →
    translateGemmaSession.layout ≠ .bdpa := by
  intro hall
  apply prefill_attention_excludes_bdpa translateGemmaSteps translateGemmaSession
  · exact ⟨{
      id := "prefill_attention",
      phase := .prefill,
      opClass := .attention,
    }, by decide, rfl, rfl⟩
  · exact hall
