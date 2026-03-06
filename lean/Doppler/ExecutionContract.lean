import Doppler.Model

structure ExecutionStep where
  id : String
  phase : Phase
  opClass : OpClass
  deriving Repr, DecidableEq

structure SessionConfig where
  layout : KVLayout
  disableCommandBatching : Bool
  decodeBatchSize : Nat
  headDim : Nat
  kvLen : Nat
  coldQuantMode : ColdQuantMode := .none
  deriving Repr, DecidableEq

def BDPA_MAX_HEAD_DIM : Nat := 256
def BDPA_MAX_KV_LEN : Nat := 2048
def TIERED_MAX_QUANT_HEAD_DIM : Nat := 256

def layoutAdmitsAttentionPhase : KVLayout → Phase → Bool
  | .bdpa, .prefill => false
  | .bdpa, .both => false
  | _, _ => true

def layoutAdmitsStep (layout : KVLayout) (step : ExecutionStep) : Bool :=
  match step.opClass with
  | .attention => layoutAdmitsAttentionPhase layout step.phase
  | _ => true

def stepCompatible (step : ExecutionStep) (session : SessionConfig) : Bool :=
  layoutAdmitsStep session.layout step

def allStepsCompatible (steps : List ExecutionStep) (session : SessionConfig) : Bool :=
  steps.all (fun step => stepCompatible step session)

def bdpaHeadDimValid (headDim : Nat) : Bool :=
  decide (headDim ≤ BDPA_MAX_HEAD_DIM)

def bdpaKVLenValid (kvLen : Nat) : Bool :=
  decide (kvLen ≤ BDPA_MAX_KV_LEN)

def tieredQuantValid (headDim : Nat) (coldQuant : ColdQuantMode) : Bool :=
  match coldQuant with
  | .none => true
  | _ => decide (headDim ≤ TIERED_MAX_QUANT_HEAD_DIM)

def sessionConsistent (session : SessionConfig) : Bool :=
  match session.layout with
  | .bdpa =>
    session.disableCommandBatching
      && decide (session.decodeBatchSize ≤ 1)
      && bdpaHeadDimValid session.headDim
      && bdpaKVLenValid session.kvLen
  | .tiered =>
    tieredQuantValid session.headDim session.coldQuantMode
  | _ => true

theorem bdpa_rejects_prefill_attention
    (step : ExecutionStep)
    (session : SessionConfig) :
    step.opClass = .attention →
    session.layout = .bdpa →
    step.phase = .prefill →
    stepCompatible step session = false := by
  intro hop hlayout hphase
  simp [stepCompatible, layoutAdmitsStep, layoutAdmitsAttentionPhase, hop, hlayout, hphase]

theorem bdpa_rejects_both_attention
    (step : ExecutionStep)
    (session : SessionConfig) :
    step.opClass = .attention →
    session.layout = .bdpa →
    step.phase = .both →
    stepCompatible step session = false := by
  intro hop hlayout hphase
  simp [stepCompatible, layoutAdmitsStep, layoutAdmitsAttentionPhase, hop, hlayout, hphase]

theorem prefill_attention_excludes_bdpa
    (steps : List ExecutionStep)
    (session : SessionConfig) :
    (∃ step, step ∈ steps ∧ step.opClass = .attention ∧ step.phase = .prefill) →
    allStepsCompatible steps session = true →
    session.layout ≠ .bdpa := by
  intro hex hall hbdpa
  rcases hex with ⟨step, hmem, hop, hphase⟩
  have hcomp : stepCompatible step session = true := by
    exact (List.all_eq_true.mp hall) step hmem
  have hreject : stepCompatible step session = false :=
    bdpa_rejects_prefill_attention step session hop hbdpa hphase
  simp [hreject] at hcomp

theorem both_attention_excludes_bdpa
    (steps : List ExecutionStep)
    (session : SessionConfig) :
    (∃ step, step ∈ steps ∧ step.opClass = .attention ∧ step.phase = .both) →
    allStepsCompatible steps session = true →
    session.layout ≠ .bdpa := by
  intro hex hall hbdpa
  rcases hex with ⟨step, hmem, hop, hphase⟩
  have hcomp : stepCompatible step session = true := by
    exact (List.all_eq_true.mp hall) step hmem
  have hreject : stepCompatible step session = false :=
    bdpa_rejects_both_attention step session hop hbdpa hphase
  simp [hreject] at hcomp

theorem bdpa_requires_no_recorder
    (session : SessionConfig) :
    session.layout = .bdpa →
    sessionConsistent session = true →
    session.disableCommandBatching = true := by
  intro hlayout hconsistent
  simp [sessionConsistent, hlayout] at hconsistent
  exact hconsistent.1.1.1

theorem bdpa_requires_single_token_decode
    (session : SessionConfig) :
    session.layout = .bdpa →
    sessionConsistent session = true →
    session.decodeBatchSize ≤ 1 := by
  intro hlayout hconsistent
  simp [sessionConsistent, hlayout] at hconsistent
  simpa using hconsistent.1.1.2

theorem bdpa_head_dim_bound
    (session : SessionConfig) :
    session.layout = .bdpa →
    sessionConsistent session = true →
    session.headDim ≤ BDPA_MAX_HEAD_DIM := by
  intro hlayout hconsistent
  simp [sessionConsistent, hlayout, bdpaHeadDimValid] at hconsistent
  exact hconsistent.1.2

theorem bdpa_kv_len_bound
    (session : SessionConfig) :
    session.layout = .bdpa →
    sessionConsistent session = true →
    session.kvLen ≤ BDPA_MAX_KV_LEN := by
  intro hlayout hconsistent
  simp [sessionConsistent, hlayout, bdpaKVLenValid] at hconsistent
  exact hconsistent.2

theorem tiered_quant_headDim_bound
    (session : SessionConfig) :
    session.layout = .tiered →
    sessionConsistent session = true →
    session.coldQuantMode ≠ .none →
    session.headDim ≤ TIERED_MAX_QUANT_HEAD_DIM := by
  intro hlayout hconsistent hquant
  cases hcold : session.coldQuantMode with
  | none =>
    exact False.elim (hquant hcold)
  | int8 =>
    simp [sessionConsistent, hlayout, tieredQuantValid, hcold] at hconsistent
    exact hconsistent
  | int4 =>
    simp [sessionConsistent, hlayout, tieredQuantValid, hcold] at hconsistent
    exact hconsistent
