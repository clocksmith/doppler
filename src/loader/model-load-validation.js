function hasExpertGroups(manifest) {
  if (!manifest?.groups) return false;
  return Object.keys(manifest.groups).some((groupId) => groupId.includes('.expert.'));
}

export function detectMoE(manifest) {
  const moeConfig = manifest.moeConfig;
  const isMoE = moeConfig != null && (moeConfig.numExperts ?? 0) > 1;
  if (!isMoE && hasExpertGroups(manifest)) {
    throw new Error(
      `Manifest "${manifest.modelId ?? 'unknown'}" missing moeConfig for MoE model. Re-convert with moeConfig.`
    );
  }
  return isMoE;
}
