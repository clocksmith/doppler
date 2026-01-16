


export function isMoELayerLocal(layerIdx, config, layerWeights) {
  if (!config.useMoE) return false;
  if (layerWeights?.routerWeight) return true;
  const layerTypes = config.layerTypes;
  if (Array.isArray(layerTypes) && layerIdx < layerTypes.length) {
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
