import { SLOT_LIMITS as VLIW_SLOT_LIMITS } from '../vliw-shared.js';
import { DEFAULT_SPEC } from './constants.js';

function cloneArray(value, fallback) {
  if (Array.isArray(value)) return value.slice();
  if (Array.isArray(fallback)) return fallback.slice();
  return [];
}

function cloneObject(value, fallback) {
  const base = value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  const source = base || fallback || {};
  const out = {};
  Object.keys(source).forEach((key) => {
    out[key] = source[key];
  });
  return out;
}

export function normalizeSpec(input = {}, options = {}) {
  const spec = { ...DEFAULT_SPEC, ...(input || {}) };
  const mode = options?.mode === 'parity' ? 'parity' : 'relaxed';
  spec.selection_mode_by_round = cloneObject(input.selection_mode_by_round, DEFAULT_SPEC.selection_mode_by_round);
  spec.cached_round_aliases = cloneObject(input.cached_round_aliases, DEFAULT_SPEC.cached_round_aliases);
  spec.cached_round_depth = cloneObject(input.cached_round_depth, DEFAULT_SPEC.cached_round_depth);
  spec.cached_round_x = cloneObject(input.cached_round_x, DEFAULT_SPEC.cached_round_x);
  spec.base_cached_rounds = cloneArray(input.base_cached_rounds, DEFAULT_SPEC.base_cached_rounds);
  spec.depth4_cached_rounds = cloneArray(input.depth4_cached_rounds, DEFAULT_SPEC.depth4_cached_rounds);
  if (spec.vector_block == null) spec.vector_block = DEFAULT_SPEC.vector_block;
  if (spec.extra_vecs == null) spec.extra_vecs = DEFAULT_SPEC.extra_vecs;
  if (mode === 'parity') {
    spec.shifts_on_valu = true;
  }
  if (mode !== 'parity' && spec.selection_mode && ['bitmask', 'mask', 'mask_precompute'].includes(spec.selection_mode)) {
    if (!Number.isFinite(spec.extra_vecs) || spec.extra_vecs <= 0) {
      spec.extra_vecs = DEFAULT_SPEC.extra_vecs;
    }
  }
  return spec;
}

export function resolveCaps(spec, options = {}) {
  const mode = options?.capsMode === 'slot_limits' ? 'slot_limits' : 'spec';
  if (mode === 'slot_limits') {
    return { ...VLIW_SLOT_LIMITS };
  }
  return {
    alu: Number.isFinite(spec.alu_cap) ? spec.alu_cap : VLIW_SLOT_LIMITS.alu,
    valu: Number.isFinite(spec.valu_cap) ? spec.valu_cap : VLIW_SLOT_LIMITS.valu,
    load: Number.isFinite(spec.load_cap) ? spec.load_cap : VLIW_SLOT_LIMITS.load,
    store: Number.isFinite(spec.store_cap) ? spec.store_cap : VLIW_SLOT_LIMITS.store,
    flow: Number.isFinite(spec.flow_cap) ? spec.flow_cap : VLIW_SLOT_LIMITS.flow,
    debug: Number.isFinite(spec.debug_cap) ? spec.debug_cap : VLIW_SLOT_LIMITS.debug,
  };
}
