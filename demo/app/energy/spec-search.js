import { buildLayout, getDefaultSpec } from '../../../src/inference/energy/vliw-generator.js';
import { energySpecEvalCache } from './cache.js';
import { buildVliwDatasetFromSpecInput, sliceVliwDataset } from './datasets.js';
import {
  mulberry32,
  stableStringify,
  cloneSpec,
  canonicalDepthForRound,
  requiredCachedNodes,
} from './utils.js';

const SPEC_SEARCH_PRESETS = {
  baseCachedRounds: {
    top4: [0, 1, 2, 3, 11, 12, 13, 14],
    skip_r3: [0, 1, 2, 11, 12, 13, 14],
    skip_r3_r13: [0, 1, 2, 11, 12, 14],
    loadbound: [0, 1, 2, 11, 12, 13],
  },
  selectionByRound: {
    none: null,
    bitmask_11_14: {
      11: 'bitmask',
      12: 'bitmask',
      13: 'bitmask',
      14: 'bitmask',
    },
    mask_precompute_11_14: {
      11: 'mask_precompute',
      12: 'mask_precompute',
      13: 'mask_precompute',
      14: 'mask_precompute',
    },
  },
};

const SPEC_SEARCH_SPACE = {
  selection_mode: ['eq', 'bitmask', 'mask', 'mask_precompute'],
  idx_shifted: [false, true],
  vector_block: [0, 4, 8, 16, 32],
  extra_vecs: [0, 1, 2, 3, 4],
  reset_on_valu: [false, true],
  shifts_on_valu: [false, true],
  cached_nodes: [null, 7, 15, 31, 63],
  base_cached_rounds: Object.keys(SPEC_SEARCH_PRESETS.baseCachedRounds),
  depth4_rounds: [0, 1],
  x4: [0, 8, 12, 15, 24, 32],
  selection_mode_by_round: Object.keys(SPEC_SEARCH_PRESETS.selectionByRound),
  cached_round_x: [null, 8, 16, 24, 32],
  offload_op1: [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600],
  offload_hash_op1: [false, true],
  offload_hash_shift: [false, true],
  offload_hash_op2: [false, true],
  offload_parity: [false, true],
  offload_node_xor: [false, true],
  node_ptr_incremental: [false, true],
  ptr_setup_engine: ['flow', 'alu'],
  setup_style: ['inline', 'packed'],
};

function countTasksByEngine(tasks) {
  const counts = {};
  tasks.forEach((task) => {
    if (!task || task.engine === 'debug') return;
    counts[task.engine] = (counts[task.engine] || 0) + 1;
  });
  return counts;
}

function lowerBoundCycles(counts, caps) {
  let lb = 0;
  Object.entries(caps || {}).forEach(([engine, cap]) => {
    if (engine === 'debug') return;
    const total = counts?.[engine] ?? 0;
    if (total) {
      lb = Math.max(lb, Math.ceil(total / Math.max(1, cap || 0)));
    }
  });
  return lb;
}

function resolveScoreMode(scoreMode) {
  if (scoreMode && scoreMode !== 'auto') return scoreMode;
  return 'graph';
}

function normalizeSpecCandidate(spec, baseSpec) {
  const out = spec;
  out.rounds = baseSpec.rounds;
  out.vectors = baseSpec.vectors;
  out.vlen = baseSpec.vlen;
  out.total_cycles = baseSpec.total_cycles;
  out.base_cached_rounds = Array.isArray(out.base_cached_rounds) ? out.base_cached_rounds.slice() : [];
  out.depth4_cached_rounds = Array.isArray(out.depth4_cached_rounds) ? out.depth4_cached_rounds.slice() : [];
  out.selection_mode_by_round = out.selection_mode_by_round && typeof out.selection_mode_by_round === 'object' && !Array.isArray(out.selection_mode_by_round)
    ? { ...out.selection_mode_by_round }
    : {};
  out.cached_round_aliases = out.cached_round_aliases && typeof out.cached_round_aliases === 'object' && !Array.isArray(out.cached_round_aliases)
    ? { ...out.cached_round_aliases }
    : {};
  out.cached_round_depth = out.cached_round_depth && typeof out.cached_round_depth === 'object' && !Array.isArray(out.cached_round_depth)
    ? { ...out.cached_round_depth }
    : {};
  out.cached_round_x = out.cached_round_x && typeof out.cached_round_x === 'object' && !Array.isArray(out.cached_round_x)
    ? { ...out.cached_round_x }
    : {};
  out.depth4_rounds = Number.isFinite(out.depth4_rounds)
    ? Math.max(0, Math.min(1, Math.round(out.depth4_rounds)))
    : 0;
  out.x4 = Number.isFinite(out.x4) ? Math.max(0, Math.round(out.x4)) : 0;
  out.x5 = Number.isFinite(out.x5) ? Math.max(0, Math.round(out.x5)) : 0;
  if (out.depth4_rounds === 0) {
    out.depth4_cached_rounds = [];
    out.x4 = 0;
  } else if (!out.depth4_cached_rounds.length) {
    out.depth4_cached_rounds = [4];
  }
  return out;
}

export function resolveBaseSpec(specInput) {
  const base = { ...getDefaultSpec(), ...(specInput || {}) };
  return normalizeSpecCandidate(base, base);
}

function applySelectionModeByRound(spec, presetKey) {
  const preset = SPEC_SEARCH_PRESETS.selectionByRound[presetKey] || null;
  spec.selection_mode_by_round = preset ? { ...preset } : {};
  if (!preset) return;
  const modes = new Set(Object.values(preset));
  if (modes.has('mask_precompute')) {
    spec.selection_mode = 'mask_precompute';
    spec.use_bitmask_selection = false;
    return;
  }
  if (modes.has('mask')) {
    spec.selection_mode = 'mask';
    spec.use_bitmask_selection = false;
    return;
  }
  if (modes.has('bitmask')) {
    spec.selection_mode = 'bitmask';
    spec.use_bitmask_selection = true;
  }
}

function applyBaseCachedRoundsPreset(spec, presetKey) {
  const preset = SPEC_SEARCH_PRESETS.baseCachedRounds[presetKey];
  spec.base_cached_rounds = preset ? preset.slice() : [];
}

function applyCachedRoundX(spec, value) {
  spec.cached_round_x = {};
  if (!Number.isFinite(value) || value <= 0) return;
  const rounds = Array.isArray(spec.base_cached_rounds) ? spec.base_cached_rounds : [];
  rounds.forEach((round) => {
    if (canonicalDepthForRound(round) == null) return;
    if (Number.parseInt(round, 10) === 4) return;
    spec.cached_round_x[round] = value;
  });
}

function choose(rng, values) {
  return values[Math.floor(rng() * values.length)];
}

function mutateSpecCandidate(baseSpec, rng, mutationCount) {
  const spec = normalizeSpecCandidate(cloneSpec(baseSpec), baseSpec);
  const mutations = Math.max(1, mutationCount);
  const mutators = [
    (s) => {
      const value = choose(rng, SPEC_SEARCH_SPACE.selection_mode);
      s.selection_mode = value;
      s.use_bitmask_selection = value === 'bitmask';
    },
    (s) => {
      s.idx_shifted = choose(rng, SPEC_SEARCH_SPACE.idx_shifted);
    },
    (s) => {
      s.vector_block = choose(rng, SPEC_SEARCH_SPACE.vector_block);
    },
    (s) => {
      s.extra_vecs = choose(rng, SPEC_SEARCH_SPACE.extra_vecs);
    },
    (s) => {
      s.reset_on_valu = choose(rng, SPEC_SEARCH_SPACE.reset_on_valu);
    },
    (s) => {
      s.shifts_on_valu = choose(rng, SPEC_SEARCH_SPACE.shifts_on_valu);
    },
    (s) => {
      s.cached_nodes = choose(rng, SPEC_SEARCH_SPACE.cached_nodes);
    },
    (s) => {
      applyBaseCachedRoundsPreset(s, choose(rng, SPEC_SEARCH_SPACE.base_cached_rounds));
    },
    (s) => {
      s.depth4_rounds = choose(rng, SPEC_SEARCH_SPACE.depth4_rounds);
      s.depth4_cached_rounds = s.depth4_rounds > 0 ? [4] : [];
      if (s.depth4_rounds === 0) s.x4 = 0;
    },
    (s) => {
      s.x4 = choose(rng, SPEC_SEARCH_SPACE.x4);
      if (!s.depth4_rounds) s.x4 = 0;
    },
    (s) => {
      applySelectionModeByRound(s, choose(rng, SPEC_SEARCH_SPACE.selection_mode_by_round));
    },
    (s) => {
      applyCachedRoundX(s, choose(rng, SPEC_SEARCH_SPACE.cached_round_x));
    },
    (s) => {
      s.offload_op1 = choose(rng, SPEC_SEARCH_SPACE.offload_op1);
    },
    (s) => {
      s.offload_hash_op1 = choose(rng, SPEC_SEARCH_SPACE.offload_hash_op1);
    },
    (s) => {
      s.offload_hash_shift = choose(rng, SPEC_SEARCH_SPACE.offload_hash_shift);
    },
    (s) => {
      s.offload_hash_op2 = choose(rng, SPEC_SEARCH_SPACE.offload_hash_op2);
    },
    (s) => {
      s.offload_parity = choose(rng, SPEC_SEARCH_SPACE.offload_parity);
    },
    (s) => {
      s.offload_node_xor = choose(rng, SPEC_SEARCH_SPACE.offload_node_xor);
    },
    (s) => {
      s.node_ptr_incremental = choose(rng, SPEC_SEARCH_SPACE.node_ptr_incremental);
    },
    (s) => {
      s.ptr_setup_engine = choose(rng, SPEC_SEARCH_SPACE.ptr_setup_engine);
    },
    (s) => {
      s.setup_style = choose(rng, SPEC_SEARCH_SPACE.setup_style);
    },
  ];
  for (let i = 0; i < mutations; i++) {
    const mutator = choose(rng, mutators);
    mutator(spec);
  }
  return normalizeSpecCandidate(spec, baseSpec);
}

function evaluateSpecConstraints(spec, constraintMode = 'parity') {
  const issues = [];
  let penalty = 0;
  let hardFail = false;
  const isRelaxed = constraintMode === 'relaxed';
  const vectors = Number.isFinite(spec.vectors) ? spec.vectors : 32;
  const depth4Rounds = Number.isFinite(spec.depth4_rounds) ? spec.depth4_rounds : 0;
  const depth4List = Array.isArray(spec.depth4_cached_rounds) ? spec.depth4_cached_rounds : [];
  const x4 = Number.isFinite(spec.x4) ? spec.x4 : 0;
  const x5 = Number.isFinite(spec.x5) ? spec.x5 : 0;

  if (depth4Rounds === 0 && x4 > 0) {
    hardFail = true;
    issues.push('x4 requires depth4_rounds');
  }
  if (x4 > vectors) {
    hardFail = true;
    issues.push('x4 exceeds vectors');
  }
  if (x5 > vectors) {
    hardFail = true;
    issues.push('x5 exceeds vectors');
  }
  if (depth4Rounds !== depth4List.length) {
    hardFail = true;
    issues.push('depth4_rounds mismatch');
  }
  if (depth4Rounds === 0 && depth4List.length) {
    hardFail = true;
    issues.push('depth4_cached_rounds requires depth4_rounds');
  }
  if (!isRelaxed && depth4Rounds > 0) {
    if (depth4List.length !== 1 || depth4List[0] !== 4) {
      hardFail = true;
      issues.push('depth4_cached_rounds must be [4] in parity mode');
    }
  }

  const cachedRoundDepth = spec.cached_round_depth || {};
  Object.keys(cachedRoundDepth).forEach((round) => {
    const canonical = canonicalDepthForRound(round);
    const value = Number.parseInt(cachedRoundDepth[round], 10);
    if (canonical == null || canonical !== value || value >= 4) {
      hardFail = true;
      issues.push(`invalid cached_round_depth for round ${round}`);
    }
  });

  const cachedRoundAliases = spec.cached_round_aliases || {};
  if (!isRelaxed) {
    Object.keys(cachedRoundAliases).forEach((aliasRound) => {
      const depth = Number.parseInt(cachedRoundAliases[aliasRound], 10);
      if (!Number.isFinite(depth) || ![0, 1, 2, 3].includes(depth)) {
        hardFail = true;
        issues.push(`cached_round_aliases depth invalid: ${aliasRound}:${cachedRoundAliases[aliasRound]}`);
      }
    });
  }

  const cachedRounds = new Set();
  const baseCached = Array.isArray(spec.base_cached_rounds) ? spec.base_cached_rounds : [];
  baseCached.forEach((round) => cachedRounds.add(Number.parseInt(round, 10)));
  Object.keys(cachedRoundDepth).forEach((round) => cachedRounds.add(Number.parseInt(round, 10)));
  Object.keys(cachedRoundAliases).forEach((round) => cachedRounds.add(Number.parseInt(round, 10)));

  const cachedRoundX = spec.cached_round_x || {};
  Object.keys(cachedRoundX).forEach((round) => {
    const roundId = Number.parseInt(round, 10);
    const canonical = canonicalDepthForRound(roundId);
    const value = Number.parseInt(cachedRoundX[round], 10);
    if (canonical == null || roundId === 4) {
      hardFail = true;
      issues.push(`invalid cached_round_x round ${round}`);
      return;
    }
    if (!Number.isFinite(value) || value < 0 || value > vectors) {
      hardFail = true;
      issues.push(`invalid cached_round_x value for round ${round}`);
      return;
    }
    if (!isRelaxed && !cachedRounds.has(roundId)) {
      hardFail = true;
      issues.push(`cached_round_x for non-cached round ${round}`);
    }
    if (isRelaxed && value <= 0) {
      hardFail = true;
      issues.push(`invalid cached_round_x value for round ${round}`);
    }
  });

  if (spec.cached_nodes != null) {
    let maxDepth = 0;
    if (isRelaxed) {
      baseCached.forEach((round) => {
        const canonical = canonicalDepthForRound(round);
        if (canonical != null && canonical > maxDepth) maxDepth = canonical;
      });
      if (depth4Rounds > 0) maxDepth = Math.max(maxDepth, 4);
      if (x5 > 0) maxDepth = Math.max(maxDepth, 5);
    } else {
      baseCached.forEach((round) => {
        const canonical = canonicalDepthForRound(round);
        if (canonical != null && canonical > maxDepth) maxDepth = canonical;
      });
      Object.values(cachedRoundDepth).forEach((depthValue) => {
        const depth = Number.parseInt(depthValue, 10);
        if (Number.isFinite(depth)) maxDepth = Math.max(maxDepth, depth);
      });
      Object.values(cachedRoundAliases).forEach((depthValue) => {
        const depth = Number.parseInt(depthValue, 10);
        if (Number.isFinite(depth)) maxDepth = Math.max(maxDepth, depth);
      });
      if (depth4Rounds > 0 && x4 > 0) maxDepth = Math.max(maxDepth, 4);
      if (x5 > 0) maxDepth = Math.max(maxDepth, 5);
    }
    const required = requiredCachedNodes(maxDepth);
    if (spec.cached_nodes < required) {
      hardFail = true;
      issues.push(`cached_nodes < ${required}`);
    }
  }

  let selectionMode = spec.selection_mode;
  if (!selectionMode) {
    selectionMode = spec.use_bitmask_selection ? 'bitmask' : 'eq';
  }
  const extraVecs = Number.isFinite(spec.extra_vecs) ? spec.extra_vecs : 0;
  if (selectionMode === 'mask_precompute') {
    if (extraVecs < 4) {
      penalty += isRelaxed ? 2 : 3;
      issues.push('mask_precompute extra_vecs < 4');
    }
    if (!spec.idx_shifted) {
      penalty += 1;
      issues.push('mask_precompute requires idx_shifted');
    }
  }
  if (selectionMode === 'bitmask') {
    const required = isRelaxed && depth4Rounds === 0 ? 1 : 3;
    if (extraVecs < required) {
      penalty += 1;
      issues.push('bitmask extra_vecs too small');
    }
  }

  const vectorBlock = Number.isFinite(spec.vector_block) ? spec.vector_block : 0;
  if (vectorBlock > 0 && vectors % vectorBlock !== 0) {
    penalty += isRelaxed ? 1 : 0.5;
    issues.push('vector_block not divisible');
  }

  const selectionByRound = spec.selection_mode_by_round || {};
  if (isRelaxed) {
    Object.keys(selectionByRound).forEach((round) => {
      if (canonicalDepthForRound(round) == null) {
        penalty += 0.5;
        issues.push(`selection_mode_by_round unused (${round})`);
      }
    });
  } else {
    for (const [roundId, mode] of Object.entries(selectionByRound)) {
      if (!['eq', 'mask', 'bitmask', 'mask_precompute'].includes(mode)) {
        penalty += 1;
        issues.push(`selection_mode_by_round invalid mode ${mode} at round ${roundId}`);
        break;
      }
      if (['mask', 'bitmask', 'mask_precompute'].includes(mode) && selectionMode === 'eq') {
        penalty += 1;
        issues.push('selection_mode_by_round requires extras but global selection_mode=eq');
        break;
      }
      if (mode === 'bitmask' && extraVecs < 3) {
        penalty += 1;
        issues.push('selection_mode_by_round bitmask with extra_vecs < 3');
        break;
      }
      if (mode === 'mask_precompute' && extraVecs < 4) {
        penalty += 1;
        issues.push('selection_mode_by_round mask_precompute with extra_vecs < 4');
        break;
      }
      if (mode === 'mask_precompute' && !spec.idx_shifted) {
        penalty += 0.5;
        issues.push('selection_mode_by_round mask_precompute with idx_shifted=0');
        break;
      }
    }
  }

  if (!isRelaxed) {
    try {
      buildLayout(spec);
    } catch (error) {
      hardFail = true;
      issues.push(`layout: ${error.message}`);
    }
  }

  return { penalty, hardFail, issues };
}

function computeOffloadPenalty(spec, offloadableCount) {
  if (!Number.isFinite(offloadableCount) || offloadableCount <= 0) return 0;
  if (!Number.isFinite(spec.offload_op1) || spec.offload_op1 <= 0) return 0;
  const ratio = spec.offload_op1 / offloadableCount;
  if (ratio <= 1) return 0;
  return (ratio - 1) * 2;
}

export function formatSpecSignature(spec) {
  const parts = [
    `sel=${spec.selection_mode || 'eq'}`,
    `idx=${spec.idx_shifted ? 1 : 0}`,
    `vb=${spec.vector_block ?? 0}`,
    `extra=${spec.extra_vecs ?? 0}`,
    `cached=${spec.cached_nodes == null ? 'auto' : spec.cached_nodes}`,
    `d4=${spec.depth4_rounds ?? 0}`,
    `x4=${spec.x4 ?? 0}`,
    `off1=${spec.offload_op1 ?? 0}`,
    `ptr=${spec.ptr_setup_engine || 'flow'}`,
    `setup=${spec.setup_style || 'inline'}`,
  ];
  return parts.join(' ');
}

export async function runVliwSpecSearch({
  pipeline,
  baseSpec,
  innerRequestBase,
  vliwSearch,
  bundleLimit,
  specSearch,
}) {
  const restarts = Number.isFinite(specSearch.restarts) ? Math.max(1, Math.floor(specSearch.restarts)) : 1;
  const steps = Number.isFinite(specSearch.steps) ? Math.max(1, Math.floor(specSearch.steps)) : 20;
  const tempStart = Number.isFinite(specSearch.temperatureStart) ? specSearch.temperatureStart : 2.0;
  const tempDecay = Number.isFinite(specSearch.temperatureDecay) ? specSearch.temperatureDecay : 0.95;
  const mutationCount = Number.isFinite(specSearch.mutationCount) ? Math.max(1, Math.floor(specSearch.mutationCount)) : 2;
  const penaltyGate = Number.isFinite(specSearch.penaltyGate) ? specSearch.penaltyGate : 2;
  const cycleLambda = Number.isFinite(specSearch.cycleLambda) ? specSearch.cycleLambda : 1.0;
  const lbPenalty = Number.isFinite(specSearch.lbPenalty) ? specSearch.lbPenalty : 0;
  const targetCycles = Number.isFinite(specSearch.targetCycles) ? specSearch.targetCycles : 0;
  const scoreModeSetting = specSearch.scoreMode || vliwSearch?.scoreMode || 'auto';
  const constraintMode = specSearch?.constraints?.mode === 'relaxed' ? 'relaxed' : 'parity';
  const fallbackBase = Number.isFinite(specSearch?.constraints?.fallbackCycles)
    ? specSearch.constraints.fallbackCycles
    : 10000;
  const effectiveBundleLimit = constraintMode === 'parity' ? 0 : bundleLimit;
  const innerSteps = Number.isFinite(specSearch.innerSteps)
    ? Math.max(1, Math.floor(specSearch.innerSteps))
    : null;
  const rng = mulberry32(Number.isFinite(specSearch.seed) ? specSearch.seed : 1337);
  const start = performance.now();
  let best = null;
  let bestEnergy = Number.POSITIVE_INFINITY;
  const candidates = [];

  async function evaluateSpec(spec) {
    const specKey = stableStringify(spec);
    const capsMode = constraintMode === 'parity' ? 'slot_limits' : 'spec';
    const schedulerPolicies = Array.isArray(vliwSearch?.schedulerPolicies) ? vliwSearch.schedulerPolicies : null;
    const schedulerRestarts = Number.isFinite(vliwSearch?.schedulerRestarts)
      ? vliwSearch.schedulerRestarts
      : null;
    const evalKey = stableStringify({
      mode: constraintMode,
      scoreMode: scoreModeSetting,
      penaltyGate,
      cycleLambda,
      lbPenalty,
      targetCycles,
      fallbackCycles: fallbackBase,
      spec: specKey,
      innerSteps: innerSteps ?? innerRequestBase?.steps ?? null,
      bundleLimit: effectiveBundleLimit ?? null,
      vliwSearch: vliwSearch ?? null,
      schedulerPolicies,
      schedulerRestarts,
      capsMode,
      initMode: innerRequestBase?.initMode ?? null,
      initScale: innerRequestBase?.initScale ?? null,
      seed: innerRequestBase?.seed ?? null,
    });
    if (energySpecEvalCache.has(evalKey)) {
      return energySpecEvalCache.get(evalKey);
    }
    const constraint = evaluateSpecConstraints(spec, constraintMode);
    let penalty = constraint.penalty;
    if (constraint.hardFail) {
      penalty = Number.POSITIVE_INFINITY;
    }
    const fallback = Math.max(
      10000,
      fallbackBase,
      Number.isFinite(spec.total_cycles) ? spec.total_cycles : 0,
    );
    if (constraint.hardFail || penalty > penaltyGate) {
      const energy = constraintMode === 'parity'
        ? penalty + cycleLambda * fallback
        : Number.POSITIVE_INFINITY;
      const payload = {
        spec,
        specKey,
        penalty,
        cycles: constraintMode === 'parity' ? fallback : Number.POSITIVE_INFINITY,
        energy,
        lb: 0,
        gap: 0,
        issues: constraint.issues,
        datasetMeta: null,
        scoreMode: scoreModeSetting,
      };
      energySpecEvalCache.set(evalKey, payload);
      return payload;
    }
    let dataset = null;
    try {
      dataset = await buildVliwDatasetFromSpecInput(spec, specKey, { mode: constraintMode, capsMode });
    } catch (error) {
      const payload = {
        spec,
        specKey,
        penalty: Number.POSITIVE_INFINITY,
        cycles: Number.POSITIVE_INFINITY,
        energy: Number.POSITIVE_INFINITY,
        lb: 0,
        gap: 0,
        issues: [`spec build failed: ${error.message}`],
        datasetMeta: null,
        scoreMode: scoreModeSetting,
      };
      energySpecEvalCache.set(evalKey, payload);
      return payload;
    }
    if (constraintMode === 'relaxed') {
      penalty += computeOffloadPenalty(spec, dataset.offloadableCount);
    }
    if (penalty > penaltyGate) {
      const energy = constraintMode === 'parity'
        ? penalty + cycleLambda * fallback
        : Number.POSITIVE_INFINITY;
      const payload = {
        spec,
        specKey,
        penalty,
        cycles: constraintMode === 'parity' ? fallback : Number.POSITIVE_INFINITY,
        energy,
        lb: 0,
        gap: 0,
        issues: constraint.issues,
        datasetMeta: {
          label: dataset.label,
          bundleCount: dataset.bundleCount,
          taskCount: dataset.taskCount,
          baselineCycles: dataset.baselineCycles,
          dagHash: dataset.dag?.hash ?? dataset.dagHash,
          dependencyModel: dataset.dependencyModel ?? null,
          spec: dataset.spec ?? null,
        },
        scoreMode: scoreModeSetting,
      };
      energySpecEvalCache.set(evalKey, payload);
      return payload;
    }
    const sliced = sliceVliwDataset(dataset, effectiveBundleLimit);
    const counts = countTasksByEngine(sliced.tasks);
    const lb = lowerBoundCycles(counts, sliced.caps);
    const gap = targetCycles > 0 ? Math.max(0, lb - targetCycles) : 0;
    const resolvedScoreMode = resolveScoreMode(scoreModeSetting);
    let cycles = Number.POSITIVE_INFINITY;
    let policy = null;
    const searchConfig = {
      ...vliwSearch,
      scoreMode: resolvedScoreMode,
      mode: constraintMode,
      capsSource: capsMode === 'slot_limits' ? 'slot_limits' : 'spec',
    };
    if (Number.isFinite(dataset.spec?.sched_seed)) {
      searchConfig.schedulerSeed = dataset.spec.sched_seed;
    }
    if (Number.isFinite(dataset.spec?.sched_jitter)) {
      searchConfig.schedulerJitter = dataset.spec.sched_jitter;
    }
    if (Number.isFinite(dataset.spec?.sched_restarts)) {
      searchConfig.schedulerRestarts = dataset.spec.sched_restarts;
    }
    let result = null;
    if (resolvedScoreMode === 'lb') {
      cycles = lb;
    } else if (resolvedScoreMode === 'bundle') {
      cycles = Number.isFinite(dataset.baselineCycles) ? dataset.baselineCycles : lb;
    } else {
      const innerRequest = {
        ...innerRequestBase,
        steps: innerSteps ?? innerRequestBase.steps,
        vliw: {
          tasks: sliced.tasks,
          caps: sliced.caps,
          dependencyModel: dataset.dependencyModel ?? null,
          search: searchConfig,
        },
      };
      result = await pipeline.generate(innerRequest);
      cycles = result?.metrics?.cycles ?? lb;
      policy = result?.schedulerPolicy ?? null;
    }
    const energy = Number.isFinite(cycles)
      ? penalty + cycleLambda * cycles + lbPenalty * gap
      : Number.POSITIVE_INFINITY;
    const payload = {
      spec,
      specKey,
      penalty,
      cycles,
      energy,
      lb,
      gap,
      issues: constraint.issues,
      datasetMeta: {
        label: dataset.label,
        bundleCount: sliced.bundleCount ?? dataset.bundleCount,
        taskCount: sliced.taskCount ?? dataset.taskCount,
        baselineCycles: dataset.baselineCycles ?? dataset.bundleCount,
        dagHash: dataset.dag?.hash ?? dataset.dagHash,
        dependencyModel: dataset.dependencyModel ?? null,
        spec: dataset.spec ?? null,
      },
      scoreMode: resolvedScoreMode,
      policy,
    };
    energySpecEvalCache.set(evalKey, payload);
    return payload;
  }

  function pushCandidate(entry) {
    if (!entry || !Number.isFinite(entry.energy)) return;
    candidates.push(entry);
    candidates.sort((a, b) => a.energy - b.energy);
    if (candidates.length > 6) {
      candidates.length = 6;
    }
  }

  for (let restart = 0; restart < restarts; restart++) {
    let current = await evaluateSpec(normalizeSpecCandidate(cloneSpec(baseSpec), baseSpec));
    let currentEnergy = current.energy;
    let temperature = tempStart;
    if (Number.isFinite(currentEnergy) && currentEnergy < bestEnergy) {
      bestEnergy = currentEnergy;
      best = current;
      pushCandidate(current);
    }
    for (let step = 0; step < steps; step++) {
      const candidateSpec = mutateSpecCandidate(current.spec, rng, mutationCount);
      const candidate = await evaluateSpec(candidateSpec);
      const candidateEnergy = candidate.energy;
      const delta = candidateEnergy - currentEnergy;
      const accept = (
        (!Number.isFinite(currentEnergy) && Number.isFinite(candidateEnergy))
        || (Number.isFinite(candidateEnergy)
          && (delta <= 0 || rng() < Math.exp(-delta / Math.max(temperature, 1e-6))))
      );
      if (accept) {
        current = candidate;
        currentEnergy = candidateEnergy;
      }
      if (Number.isFinite(candidateEnergy) && candidateEnergy < bestEnergy) {
        bestEnergy = candidateEnergy;
        best = candidate;
      }
      pushCandidate(candidate);
      temperature *= tempDecay;
    }
  }

  if (!best || !Number.isFinite(best.energy)) {
    throw new Error('Spec search failed to find a valid configuration.');
  }

  const capsMode = constraintMode === 'parity' ? 'slot_limits' : 'spec';
  const finalDataset = await buildVliwDatasetFromSpecInput(best.spec, best.specKey, {
    mode: constraintMode,
    capsMode,
    includeOps: true,
  });
  const sliced = sliceVliwDataset(finalDataset, effectiveBundleLimit);
  const finalScoreMode = best.scoreMode || resolveScoreMode(scoreModeSetting);
  const finalSearchConfig = {
    ...vliwSearch,
    scoreMode: finalScoreMode,
    mode: constraintMode,
    capsSource: capsMode === 'slot_limits' ? 'slot_limits' : 'spec',
  };
  if (Number.isFinite(finalDataset.spec?.sched_seed)) {
    finalSearchConfig.schedulerSeed = finalDataset.spec.sched_seed;
  }
  if (Number.isFinite(finalDataset.spec?.sched_jitter)) {
    finalSearchConfig.schedulerJitter = finalDataset.spec.sched_jitter;
  }
  if (Number.isFinite(finalDataset.spec?.sched_restarts)) {
    finalSearchConfig.schedulerRestarts = finalDataset.spec.sched_restarts;
  }
  const finalRequest = {
    ...innerRequestBase,
    vliw: {
      tasks: sliced.tasks,
      caps: sliced.caps,
      dependencyModel: finalDataset.dependencyModel ?? null,
      search: finalSearchConfig,
    },
  };
  const finalResult = await pipeline.generate(finalRequest);

  return {
    bestSpec: best.spec,
    bestEnergy: best.energy,
    bestPenalty: best.penalty,
    bestCycles: best.cycles,
    candidates: candidates.slice(),
    dataset: finalDataset,
    sliced,
    result: finalResult,
    totalMs: performance.now() - start,
    restarts,
    steps,
    cycleLambda,
    lbPenalty,
    targetCycles,
    penaltyGate,
    fallbackCycles: Math.max(
      10000,
      fallbackBase,
      Number.isFinite(best?.spec?.total_cycles) ? best.spec.total_cycles : 0,
    ),
    innerSteps,
    constraintMode,
    scheduler: finalResult?.scheduler ?? null,
    scoreMode: finalScoreMode,
  };
}
