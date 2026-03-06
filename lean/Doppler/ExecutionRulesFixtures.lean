import Doppler.ExecutionRules

def executionRulesChecks : List (String × Bool) := [
  ("execution_rules_decode_recorder_happy_path",
    decodeRecorderEnabled
      {
        hasDevice := true,
        debug := false,
        disableCommandBatching := false,
        kvLayout := .other
      }),
  ("execution_rules_decode_recorder_rejects_bdpa_paged",
    decodeRecorderEnabled
      {
        hasDevice := true,
        debug := false,
        disableCommandBatching := false,
        kvLayout := .bdpa_paged
      }),
  ("execution_rules_batch_decode_happy_path",
    batchDecodeEnabled
      {
        batchSize := 2,
        useGPU := true,
        gpuSamplingAvailable := true,
        disableMultiTokenDecode := false,
        disableCommandBatching := false,
        isBdpaPagedLayout := false,
        finitenessFallbackWindowOpen := false
      }),
  ("execution_rules_batch_decode_rejects_batch_one",
    batchDecodeEnabled
      {
        batchSize := 1,
        useGPU := true,
        gpuSamplingAvailable := true,
        disableMultiTokenDecode := false,
        disableCommandBatching := false,
        isBdpaPagedLayout := false,
        finitenessFallbackWindowOpen := false
      }),
  ("execution_rules_batch_decode_rejects_finiteness_fallback",
    batchDecodeEnabled
      {
        batchSize := 2,
        useGPU := true,
        gpuSamplingAvailable := true,
        disableMultiTokenDecode := false,
        disableCommandBatching := false,
        isBdpaPagedLayout := false,
        finitenessFallbackWindowOpen := true
      })
]
