inductive ExecutionRulesKVLayout where
  | bdpa_paged
  | other
  deriving Repr, DecidableEq

structure DecodeRecorderContext where
  hasDevice : Bool
  debug : Bool
  disableCommandBatching : Bool
  kvLayout : ExecutionRulesKVLayout
  deriving Repr, DecidableEq

def decodeRecorderEnabled (context : DecodeRecorderContext) : Bool :=
  context.hasDevice
    && !context.debug
    && !context.disableCommandBatching
    && context.kvLayout != .bdpa_paged

structure BatchDecodeContext where
  batchSize : Nat
  useGPU : Bool
  gpuSamplingAvailable : Bool
  disableMultiTokenDecode : Bool
  disableCommandBatching : Bool
  isBdpaPagedLayout : Bool
  finitenessFallbackWindowOpen : Bool
  deriving Repr, DecidableEq

def batchDecodeEnabled (context : BatchDecodeContext) : Bool :=
  context.batchSize > 1
    && context.useGPU
    && context.gpuSamplingAvailable
    && !context.disableMultiTokenDecode
    && !context.disableCommandBatching
    && !context.isBdpaPagedLayout
    && !context.finitenessFallbackWindowOpen

theorem decode_recorder_enabled_on_happy_path :
    decodeRecorderEnabled
      {
        hasDevice := true,
        debug := false,
        disableCommandBatching := false,
        kvLayout := .other
      } = true := by
  decide

theorem decode_recorder_requires_device
    (debug disableCommandBatching : Bool)
    (kvLayout : ExecutionRulesKVLayout) :
    decodeRecorderEnabled
      {
        hasDevice := false,
        debug := debug,
        disableCommandBatching := disableCommandBatching,
        kvLayout := kvLayout
      } = false := by
  cases debug <;> cases disableCommandBatching <;> cases kvLayout <;> decide

theorem decode_recorder_rejects_bdpa_paged
    (hasDevice debug disableCommandBatching : Bool) :
    decodeRecorderEnabled
      {
        hasDevice := hasDevice,
        debug := debug,
        disableCommandBatching := disableCommandBatching,
        kvLayout := .bdpa_paged
      } = false := by
  cases hasDevice <;> cases debug <;> cases disableCommandBatching <;> decide

theorem batch_decode_enabled_on_happy_path :
    batchDecodeEnabled
      {
        batchSize := 2,
        useGPU := true,
        gpuSamplingAvailable := true,
        disableMultiTokenDecode := false,
        disableCommandBatching := false,
        isBdpaPagedLayout := false,
        finitenessFallbackWindowOpen := false
      } = true := by
  decide

theorem batch_decode_requires_batch_gt_one
    (useGPU gpuSamplingAvailable disableMultiTokenDecode disableCommandBatching isBdpaPagedLayout finitenessFallbackWindowOpen : Bool) :
    batchDecodeEnabled
      {
        batchSize := 1,
        useGPU := useGPU,
        gpuSamplingAvailable := gpuSamplingAvailable,
        disableMultiTokenDecode := disableMultiTokenDecode,
        disableCommandBatching := disableCommandBatching,
        isBdpaPagedLayout := isBdpaPagedLayout,
        finitenessFallbackWindowOpen := finitenessFallbackWindowOpen
      } = false := by
  cases useGPU <;>
    cases gpuSamplingAvailable <;>
    cases disableMultiTokenDecode <;>
    cases disableCommandBatching <;>
    cases isBdpaPagedLayout <;>
    cases finitenessFallbackWindowOpen <;>
    decide

theorem batch_decode_rejects_command_batching_disabled
    (batchSize : Nat)
    (useGPU gpuSamplingAvailable disableMultiTokenDecode isBdpaPagedLayout finitenessFallbackWindowOpen : Bool) :
    batchDecodeEnabled
      {
        batchSize := batchSize,
        useGPU := useGPU,
        gpuSamplingAvailable := gpuSamplingAvailable,
        disableMultiTokenDecode := disableMultiTokenDecode,
        disableCommandBatching := true,
        isBdpaPagedLayout := isBdpaPagedLayout,
        finitenessFallbackWindowOpen := finitenessFallbackWindowOpen
      } = false := by
  simp [batchDecodeEnabled]

theorem batch_decode_rejects_finiteness_fallback
    (batchSize : Nat)
    (useGPU gpuSamplingAvailable disableMultiTokenDecode disableCommandBatching isBdpaPagedLayout : Bool) :
    batchDecodeEnabled
      {
        batchSize := batchSize,
        useGPU := useGPU,
        gpuSamplingAvailable := gpuSamplingAvailable,
        disableMultiTokenDecode := disableMultiTokenDecode,
        disableCommandBatching := disableCommandBatching,
        isBdpaPagedLayout := isBdpaPagedLayout,
        finitenessFallbackWindowOpen := true
      } = false := by
  simp [batchDecodeEnabled]
