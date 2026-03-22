
// Failure reduction hooks for operator-level differential debugging.
//
// When the diff engine identifies a failure (divergent operator),
// these hooks shrink the failure to a minimal reproducible case:
//
//   prompt    — truncate prompt to shortest length that reproduces
//   tokens    — limit generated token count
//   graph     — isolate to a layer range or operator subset
//
// Reducer modes are exposed through the diagnose command surface.
// The goal: smaller shareable forensic bundles.

// ============================================================================
// Reducer Modes
// ============================================================================

export const REDUCER_MODES = Object.freeze({
  PROMPT: 'prompt',
  TOKENS: 'tokens',
  GRAPH: 'graph',
});

// ============================================================================
// Reduction Config
// ============================================================================

export function createReductionConfig(options = {}) {
  return {
    mode: options.mode ?? null,
    enabled: options.enabled === true,

    prompt: {
      minLength: options.promptMinLength ?? 1,
      strategy: options.promptStrategy ?? 'binary_search',
    },

    tokens: {
      maxTokens: options.maxTokens ?? 1,
      step: options.tokenStep ?? 1,
    },

    graph: {
      startLayer: options.startLayer ?? null,
      endLayer: options.endLayer ?? null,
      targetOpIds: options.targetOpIds ?? [],
      targetStages: options.targetStages ?? [],
    },

    divergenceOpId: options.divergenceOpId ?? null,
  };
}

// ============================================================================
// Prompt Reduction
// ============================================================================

export function computePromptReductionSteps(tokenIds, config) {
  const minLen = config.prompt.minLength;
  const strategy = config.prompt.strategy;
  const totalLen = tokenIds.length;

  if (totalLen <= minLen) return [tokenIds.slice()];

  if (strategy === 'linear') {
    return computeLinearSteps(tokenIds, minLen);
  }

  return computeBinarySearchSteps(tokenIds, minLen);
}

function computeBinarySearchSteps(tokenIds, minLen) {
  const steps = [];
  let lo = minLen;
  let hi = tokenIds.length;

  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    steps.push(tokenIds.slice(0, mid));
    hi = mid;
  }

  if (steps.length === 0 || steps[steps.length - 1].length !== minLen) {
    steps.push(tokenIds.slice(0, minLen));
  }

  return steps;
}

function computeLinearSteps(tokenIds, minLen) {
  const steps = [];
  for (let len = tokenIds.length; len >= minLen; len = Math.floor(len / 2)) {
    steps.push(tokenIds.slice(0, len));
    if (len === minLen) break;
  }
  if (steps[steps.length - 1].length !== minLen) {
    steps.push(tokenIds.slice(0, minLen));
  }
  return steps;
}

// ============================================================================
// Token Count Reduction
// ============================================================================

export function computeTokenReductionPlan(originalMaxTokens, config) {
  const step = config.tokens.step;
  const minTokens = config.tokens.maxTokens;
  const plan = [];

  for (let n = originalMaxTokens; n >= minTokens; n -= step) {
    plan.push(n);
    if (n === minTokens) break;
  }

  if (plan.length === 0 || plan[plan.length - 1] !== minTokens) {
    plan.push(minTokens);
  }

  return plan;
}

// ============================================================================
// Graph Slice
// ============================================================================

export function createGraphSlice(config) {
  const startLayer = config.graph.startLayer;
  const endLayer = config.graph.endLayer;
  const targetOpIds = config.graph.targetOpIds;
  const targetStages = config.graph.targetStages;

  return {
    startLayer,
    endLayer,
    targetOpIds,
    targetStages,

    shouldProcessLayer(layerIdx) {
      if (startLayer !== null && layerIdx < startLayer) return false;
      if (endLayer !== null && layerIdx > endLayer) return false;
      return true;
    },

    shouldProcessOp(opId, stageName) {
      if (targetOpIds.length > 0 && !targetOpIds.includes(opId)) {
        return false;
      }
      if (targetStages.length > 0 && !targetStages.includes(stageName)) {
        return false;
      }
      return true;
    },

    isSliced() {
      return startLayer !== null
        || endLayer !== null
        || targetOpIds.length > 0
        || targetStages.length > 0;
    },
  };
}

// ============================================================================
// Reduction Report
// ============================================================================

export function createReductionReport(options = {}) {
  return {
    mode: options.mode ?? null,
    originalSize: options.originalSize ?? null,
    reducedSize: options.reducedSize ?? null,
    stepsAttempted: options.stepsAttempted ?? 0,
    divergenceReproduced: options.divergenceReproduced ?? false,
    divergenceOpId: options.divergenceOpId ?? null,
    minimalPromptLength: options.minimalPromptLength ?? null,
    minimalTokenCount: options.minimalTokenCount ?? null,
    graphSlice: options.graphSlice ?? null,
  };
}
