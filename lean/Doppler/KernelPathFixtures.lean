import Doppler.KernelPath

def validAliasRegistry : List KernelPathRegistryEntry := [
  { id := "gemma3-f16-fused-f16a-online", aliasOf := none, hasFile := true },
  { id := "gemma3-f16-fused-f32a-online", aliasOf := none, hasFile := true },
  { id := "gemma3-fallback-alias", aliasOf := some "gemma3-f16-fused-f32a-online", hasFile := false }
]

def conflictingAliasRegistry : List KernelPathRegistryEntry := [
  { id := "cycle-a", aliasOf := some "cycle-b", hasFile := false },
  { id := "cycle-b", aliasOf := some "cycle-a", hasFile := false }
]

def validFallbackPairs : List (Dtype × Dtype) := [
  (.f16, .f32),
  (.f32, .f32)
]

def conflictingFallbackPairs : List (Dtype × Dtype) := [
  (.f16, .f16),
  (.f32, .f16)
]

def kernelPathChecks : List (String × Bool) := [
  ("kernelpath_valid_aliases", registryAliasAcyclic validAliasRegistry),
  ("kernelpath_conflicting_aliases", registryAliasAcyclic conflictingAliasRegistry),
  ("kernelpath_valid_fallback_pairs", validFallbackPairs.all (fun pair => fallbackPairValid pair.1 pair.2)),
  ("kernelpath_conflicting_fallback_pairs", conflictingFallbackPairs.all (fun pair => fallbackPairValid pair.1 pair.2))
]
