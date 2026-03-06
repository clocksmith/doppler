import Doppler.Model

structure ExecutionV0GraphStep where
  phase : Phase
  src : String
  dst : String
  inputDtype : Dtype
  outputDtype : Dtype
  deriving Repr, DecidableEq

abbrev ExecutionV0SlotMap := List (String × Dtype)

def executionV0SlotLookup : ExecutionV0SlotMap → String → Option Dtype
  | [], _ => none
  | (name, dtype) :: rest, slot =>
    if name = slot then some dtype else executionV0SlotLookup rest slot

def executionV0SlotInsert
    (slots : ExecutionV0SlotMap)
    (slot : String)
    (dtype : Dtype) : ExecutionV0SlotMap :=
  (slot, dtype) :: slots.filter (fun entry => entry.1 ≠ slot)

def executionV0GraphStepCompatible
    (slots : ExecutionV0SlotMap)
    (step : ExecutionV0GraphStep) : Bool :=
  executionV0SlotLookup slots step.src = some step.inputDtype

def executionV0ApplyGraphStep
    (slots : ExecutionV0SlotMap)
    (step : ExecutionV0GraphStep) : ExecutionV0SlotMap :=
  executionV0SlotInsert slots step.dst step.outputDtype

def runExecutionV0PhaseGraph :
    ExecutionV0SlotMap → List ExecutionV0GraphStep → Option ExecutionV0SlotMap
  | slots, [] => some slots
  | slots, step :: rest =>
    if executionV0GraphStepCompatible slots step then
      runExecutionV0PhaseGraph (executionV0ApplyGraphStep slots step) rest
    else
      none

def executionV0PhaseBoundaryCompatible
    (prefillSlots : ExecutionV0SlotMap)
    (decodeStep : ExecutionV0GraphStep) : Bool :=
  match executionV0SlotLookup prefillSlots decodeStep.src with
  | some carried => carried = decodeStep.inputDtype
  | none => true

theorem producer_then_consumer_valid
    (state : Dtype) :
    runExecutionV0PhaseGraph
      [("state", state)]
      [
        { phase := .prefill, src := "state", dst := "tmp", inputDtype := state, outputDtype := .f32 },
        { phase := .prefill, src := "tmp", dst := "state", inputDtype := .f32, outputDtype := .f16 }
      ] ≠ none := by
  cases state <;> simp [
    runExecutionV0PhaseGraph,
    executionV0GraphStepCompatible,
    executionV0SlotLookup,
    executionV0ApplyGraphStep,
    executionV0SlotInsert
  ]

theorem missing_slot_rejected :
    runExecutionV0PhaseGraph
      [("state", .f16)]
      [
        { phase := .prefill, src := "tmp", dst := "state", inputDtype := .f16, outputDtype := .f16 }
      ] = none := by
  simp [runExecutionV0PhaseGraph, executionV0GraphStepCompatible, executionV0SlotLookup]

theorem phase_boundary_mismatch_rejected :
    executionV0PhaseBoundaryCompatible
      [("state", .f32)]
      { phase := .decode, src := "state", dst := "state", inputDtype := .f16, outputDtype := .f16 } = false := by
  simp [executionV0PhaseBoundaryCompatible, executionV0SlotLookup]

theorem phase_boundary_match_accepts :
    executionV0PhaseBoundaryCompatible
      [("state", .f32)]
      { phase := .decode, src := "state", dst := "state", inputDtype := .f32, outputDtype := .f16 } = true := by
  simp [executionV0PhaseBoundaryCompatible, executionV0SlotLookup]
