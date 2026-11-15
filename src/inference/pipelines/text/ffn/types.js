


// The single MoE-layer test. layerTypes names the attention mixer of each layer
// (full_attention, linear_attention, ...); it names the FFN kind only when the
// manifest declares explicit 'moe' entries. Otherwise every layer of a useMoE model
// is an MoE layer, and a missing router is a load error, not a dense fallback.
export function isMoELayer(layerIdx, config) {
  if (!config.useMoE) return false;
  const layerTypes = config.layerTypes;
  if (Array.isArray(layerTypes) && layerTypes.includes('moe')) {
    return layerTypes[layerIdx] === 'moe';
  }
  return true;
}

// Track if we've logged one-time messages
let loggedFusedDownNorm = false;


export function hasLoggedFusedDownNorm() {
  return loggedFusedDownNorm;
}


export function setLoggedFusedDownNorm(value) {
  loggedFusedDownNorm = value;
}
