import { MoERouter } from "../../../moe-router.js";
import { log } from "../../../../debug/index.js";

// ============================================================================
// MoE Router Setup
// ============================================================================

export function initMoERouter(modelConfig, moeRoutingConfig, layerWeights) {
  if (!modelConfig.useMoE) return null;

  const router = new MoERouter({
    numExperts: modelConfig.numExperts,
    topK: modelConfig.moeTopK,
    hiddenSize: modelConfig.hiddenSize,
    normalizeWeights: moeRoutingConfig.normalizeWeights,
  });

  // Find first layer with router weights
  for (let l = 0; l < modelConfig.numLayers; l++) {
    const weights = layerWeights.get(`layer_${l}`);
    if (weights?.routerWeight) {
      router.loadWeights(
        weights.routerWeight,
        weights.routerBias || null,
        weights.routerScale || null,
        weights.routerPerExpertScale || null
      );
      log.debug('Pipeline', `Loaded MoE router from layer ${l}${weights.routerBias ? ' (with bias)' : ''}`);
      break;
    }
  }

  return router;
}
