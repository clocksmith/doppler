import Doppler.ExecutionV0Graph

def executionV0GraphChecks : List (String × Bool) := [
  ("executionv0_graph_producer_then_consumer_valid",
    (runExecutionV0PhaseGraph
      [("state", .f16)]
      [
        { phase := .prefill, src := "state", dst := "tmp", inputDtype := .f16, outputDtype := .f32 },
        { phase := .prefill, src := "tmp", dst := "state", inputDtype := .f32, outputDtype := .f16 }
      ]).isSome),
  ("executionv0_graph_missing_slot_rejected",
    (runExecutionV0PhaseGraph
      [("state", .f16)]
      [
        { phase := .prefill, src := "tmp", dst := "state", inputDtype := .f16, outputDtype := .f16 }
      ]).isSome),
  ("executionv0_graph_phase_boundary_match_accepts",
    executionV0PhaseBoundaryCompatible
      [("state", .f32)]
      { phase := .decode, src := "state", dst := "state", inputDtype := .f32, outputDtype := .f16 }),
  ("executionv0_graph_phase_boundary_mismatch_rejected",
    executionV0PhaseBoundaryCompatible
      [("state", .f32)]
      { phase := .decode, src := "state", dst := "state", inputDtype := .f16, outputDtype := .f16 })
]
