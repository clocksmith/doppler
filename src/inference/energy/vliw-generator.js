import { SLOT_LIMITS as VLIW_SLOT_LIMITS, VLEN as VLIW_VLEN } from './vliw-shared.js';

const HASH_STAGES = [
  ['+', 0x7ED55D16, '+', '<<', 12],
  ['^', 0xC761C23C, '^', '>>', 19],
  ['+', 0x165667B1, '+', '<<', 5],
  ['+', 0xD3A2646C, '^', '<<', 9],
  ['+', 0xFD7046C5, '+', '<<', 3],
  ['^', 0xB55A4F09, '^', '>>', 16],
];

const DEFAULT_SPEC = {
  rounds: 16,
  vectors: 32,
  vlen: 8,
  depth4_rounds: 1,
  x4: 15,
  x5: 0,
  flow_setup: 64,
  reset_on_valu: true,
  shifts_on_valu: true,
  offload_op1: 0,
  offload_hash_op1: true,
  offload_hash_shift: false,
  offload_hash_op2: false,
  offload_parity: false,
  offload_node_xor: false,
  use_bitmask_selection: false,
  selection_mode: 'eq',
  selection_mode_by_round: {},
  valu_select: false,
  node_ptr_incremental: false,
  idx_shifted: false,
  ptr_setup_engine: 'flow',
  setup_style: 'inline',
  include_setup: true,
  proof_assume_shifted_input: false,
  proof_reset_single_op: false,
  proof_skip_const_zero: false,
  valu_pad_cycles: 0,
  vector_block: 32,
  extra_vecs: 2,
  cached_nodes: null,
  base_cached_rounds: [0, 1, 2, 3, 11, 12, 13, 14],
  depth4_cached_rounds: [4],
  cached_round_aliases: {},
  cached_round_depth: {},
  cached_round_x: {},
  valu_cap: 6,
  alu_cap: 12,
  flow_cap: 1,
  load_cap: 2,
  store_cap: 2,
  total_cycles: 1312,
  sched_seed: 0,
  sched_jitter: 0,
  sched_restarts: 1,
};

const DEFAULT_DEPENDENCY_MODEL = {
  includes_raw: true,
  includes_waw: true,
  includes_war: true,
  temp_hazard_tags: true,
  read_after_read: false,
  latency: { default: 1 },
};

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

function normalizeSpec(input = {}) {
  const spec = { ...DEFAULT_SPEC, ...(input || {}) };
  spec.selection_mode_by_round = cloneObject(input.selection_mode_by_round, DEFAULT_SPEC.selection_mode_by_round);
  spec.cached_round_aliases = cloneObject(input.cached_round_aliases, DEFAULT_SPEC.cached_round_aliases);
  spec.cached_round_depth = cloneObject(input.cached_round_depth, DEFAULT_SPEC.cached_round_depth);
  spec.cached_round_x = cloneObject(input.cached_round_x, DEFAULT_SPEC.cached_round_x);
  spec.base_cached_rounds = cloneArray(input.base_cached_rounds, DEFAULT_SPEC.base_cached_rounds);
  spec.depth4_cached_rounds = cloneArray(input.depth4_cached_rounds, DEFAULT_SPEC.depth4_cached_rounds);
  if (spec.vector_block == null) spec.vector_block = DEFAULT_SPEC.vector_block;
  if (spec.extra_vecs == null) spec.extra_vecs = DEFAULT_SPEC.extra_vecs;
  if (spec.selection_mode && ['bitmask', 'mask', 'mask_precompute'].includes(spec.selection_mode)) {
    if (!Number.isFinite(spec.extra_vecs) || spec.extra_vecs <= 0) {
      spec.extra_vecs = DEFAULT_SPEC.extra_vecs;
    }
  }
  return spec;
}

function resolveCaps(spec) {
  return {
    alu: Number.isFinite(spec.alu_cap) ? spec.alu_cap : VLIW_SLOT_LIMITS.alu,
    valu: Number.isFinite(spec.valu_cap) ? spec.valu_cap : VLIW_SLOT_LIMITS.valu,
    load: Number.isFinite(spec.load_cap) ? spec.load_cap : VLIW_SLOT_LIMITS.load,
    store: Number.isFinite(spec.store_cap) ? spec.store_cap : VLIW_SLOT_LIMITS.store,
    flow: Number.isFinite(spec.flow_cap) ? spec.flow_cap : VLIW_SLOT_LIMITS.flow,
  };
}

class Op {
  constructor(engine, slot, offloadable = false, meta = null) {
    this.engine = engine;
    this.slot = slot;
    this.offloadable = offloadable;
    this.meta = meta;
    this.id = -1;
  }
}

class ScratchAlloc {
  constructor(limit = 1536) {
    this.ptr = 0;
    this.limit = limit;
    this.map = {};
  }

  alloc(name, length = 1) {
    const addr = this.ptr;
    this.map[name] = addr;
    this.ptr += length;
    if (this.ptr > this.limit) {
      throw new Error(`scratch overflow: ${this.ptr} > ${this.limit}`);
    }
    return addr;
  }
}

function buildLayout(spec) {
  const scratch = new ScratchAlloc();
  const nVecs = spec.vectors;
  const val = Array.from({ length: nVecs }, (_, i) => scratch.alloc(`val_${i}`, VLIW_VLEN));
  const idx = Array.from({ length: nVecs }, (_, i) => scratch.alloc(`idx_${i}`, VLIW_VLEN));
  const tmp = Array.from({ length: nVecs }, (_, i) => scratch.alloc(`tmp_${i}`, VLIW_VLEN));
  const tmp2 = Array.from({ length: nVecs }, (_, i) => scratch.alloc(`tmp2_${i}`, VLIW_VLEN));
  const sel = tmp2;

  let extra = [];
  let selectionMode = spec.selection_mode;
  if (!selectionMode) {
    selectionMode = spec.use_bitmask_selection ? 'bitmask' : 'eq';
  }
  if (['bitmask', 'mask', 'mask_precompute'].includes(selectionMode)) {
    const extraVecs = spec.extra_vecs ?? 1;
    extra = Array.from({ length: extraVecs }, (_, i) => scratch.alloc(`extra_${i}`, VLIW_VLEN));
  }

  const idx_ptr = Array.from({ length: nVecs }, (_, i) => scratch.alloc(`idx_ptr_${i}`));
  const val_ptr = Array.from({ length: nVecs }, (_, i) => scratch.alloc(`val_ptr_${i}`));

  const forest_values_p = scratch.alloc('forest_values_p');
  const forest_values_v = scratch.alloc('forest_values_v', VLIW_VLEN);
  const inp_indices_p = scratch.alloc('inp_indices_p');
  const inp_values_p = scratch.alloc('inp_values_p');
  const node_tmp = scratch.alloc('node_tmp');

  let node_cache = spec.cached_nodes;
  if (node_cache == null) {
    node_cache = 31;
    if ((spec.depth4_rounds ?? 0) === 0 && (spec.x5 ?? 0) === 0) {
      node_cache = 15;
    }
  }
  const node_v = Array.from({ length: node_cache }, (_, i) => scratch.alloc(`node_v_${i}`, VLIW_VLEN));

  const const_s = {};
  const const_v = {};

  const reserveConst = (val) => {
    if (const_s[val] == null) {
      const_s[val] = scratch.alloc(`const_${val}`);
    }
    return const_s[val];
  };

  const reserveVConst = (val) => {
    if (const_v[val] == null) {
      const_v[val] = scratch.alloc(`vconst_${val}`, VLIW_VLEN);
    }
    return const_v[val];
  };

  const baseConsts = new Set([0, 1, 2, 4, 5, 6, 8, 10, 12, 14, 31, VLIW_VLEN]);
  if (spec.use_bitmask_selection) {
    baseConsts.add(11);
    baseConsts.add(13);
  }
  Array.from(baseConsts).sort((a, b) => a - b).forEach(reserveConst);

  const vecConsts = new Set([1, 2]);
  if (!spec.reset_on_valu && !spec.idx_shifted) {
    vecConsts.add(0);
  }
  if (selectionMode === 'mask' || selectionMode === 'mask_precompute') {
    vecConsts.add(3);
  }
  HASH_STAGES.forEach(([op1, val1, op2, op3, val3]) => {
    if (op1 === '+' && op2 === '+') {
      const mult = (1 + (1 << val3)) % (2 ** 32);
      vecConsts.add(mult);
      vecConsts.add(val1);
    } else {
      vecConsts.add(val1);
      vecConsts.add(val3);
    }
  });
  Array.from(vecConsts).sort((a, b) => a - b).forEach((val) => {
    reserveConst(val);
    reserveVConst(val);
  });

  if (!spec.node_ptr_incremental) {
    const nodeConstMax = node_cache + (spec.idx_shifted ? 1 : 0);
    for (let v = 0; v < nodeConstMax; v++) {
      reserveConst(v);
    }
  }

  const useBitmask = spec.use_bitmask_selection;
  const depth4Rounds = spec.depth4_rounds ?? 0;
  const x4 = spec.x4 ?? 0;
  const depth4Bitmask = useBitmask && (spec.extra_vecs ?? 1) >= 3;

  if (!useBitmask) {
    for (let v = 1; v < 15; v++) {
      reserveConst(v);
    }
  }
  if (depth4Rounds && x4 > 0 && !depth4Bitmask) {
    for (let v = 15; v < 31; v++) {
      reserveConst(v);
    }
  }
  if (depth4Bitmask && depth4Rounds && x4 > 0) {
    [17, 19, 21, 23, 25, 27, 29].forEach((v) => {
      reserveConst(v);
      if (spec.idx_shifted) {
        reserveConst(v + 1);
      }
    });
  }

  return {
    val,
    idx,
    tmp,
    tmp2,
    sel,
    extra,
    idx_ptr,
    val_ptr,
    node_v,
    forest_values_p,
    forest_values_v,
    inp_indices_p,
    inp_values_p,
    node_tmp,
    const_s,
    const_v,
  };
}

let ORDERED_OPS = null;
let SEQ = 0;
let USE_VALU_SELECT = false;
let VALU_OPS_REF = null;

function recordOp(op) {
  if (!ORDERED_OPS) return;
  SEQ += 1;
  if (!op.meta) op.meta = {};
  op.meta._seq = SEQ;
  ORDERED_OPS.push(op);
}

function tagTemp(meta, key) {
  const next = meta ? { ...meta } : {};
  let temps = next.temp;
  if (!temps) {
    temps = [];
  } else if (typeof temps === 'string') {
    temps = [temps];
  } else {
    temps = temps.slice();
  }
  temps.push(key);
  next.temp = temps;
  return next;
}

function splitHashStages() {
  const linear = [];
  const bitwise = [];
  HASH_STAGES.forEach(([op1, val1, op2, op3, val3]) => {
    if (op1 === '+' && op2 === '+') {
      const mult = (1 + (1 << val3)) % (2 ** 32);
      linear.push({ mult, add: val1 });
    } else {
      bitwise.push({ op1, const: val1, op2, shift_op: op3, shift: val3 });
    }
  });
  return { linear, bitwise };
}

function vaddr(base) {
  return Array.from({ length: VLIW_VLEN }, (_, i) => base + i);
}

function addValu(list, op, dest, a, b, meta, offloadable = false) {
  const newOp = new Op('valu', [op, dest, a, b], offloadable, meta || null);
  recordOp(newOp);
  list.push(newOp);
}

function addVmuladd(list, dest, a, b, c, meta) {
  const newOp = new Op('valu', ['multiply_add', dest, a, b, c], false, meta || null);
  recordOp(newOp);
  list.push(newOp);
}

function addVselect(list, dest, cond, a, b, meta) {
  if (USE_VALU_SELECT && VALU_OPS_REF && dest !== b) {
    addValu(VALU_OPS_REF, '-', dest, a, b, meta);
    addVmuladd(VALU_OPS_REF, dest, cond, dest, b, meta);
    return;
  }
  const newOp = new Op('flow', ['vselect', dest, cond, a, b], false, meta || null);
  recordOp(newOp);
  list.push(newOp);
}

function addFlowAddImm(list, dest, a, imm, meta) {
  const newOp = new Op('flow', ['add_imm', dest, a, imm], false, meta || null);
  recordOp(newOp);
  list.push(newOp);
}

function addVbroadcast(list, dest, src, meta) {
  const newOp = new Op('valu', ['vbroadcast', dest, src], false, meta || null);
  recordOp(newOp);
  list.push(newOp);
}

function addLoad(list, dest, addr, meta) {
  const newOp = new Op('load', ['load', dest, addr], false, meta || null);
  recordOp(newOp);
  list.push(newOp);
}

function addConst(list, dest, val, meta) {
  const newOp = new Op('load', ['const', dest, val], false, meta || null);
  recordOp(newOp);
  list.push(newOp);
}

function addAluVec(list, op, dest, a, bScalar, meta) {
  for (let lane = 0; lane < VLIW_VLEN; lane++) {
    const newOp = new Op('alu', [op, dest + lane, a + lane, bScalar], false, meta || null);
    recordOp(newOp);
    list.push(newOp);
  }
}

function addAlu(list, op, dest, a, b, meta) {
  const newOp = new Op('alu', [op, dest, a, b], false, meta || null);
  recordOp(newOp);
  list.push(newOp);
}

function idxConst(spec, const_s, val) {
  if (spec.idx_shifted) {
    return const_s[val + 1];
  }
  return const_s[val];
}

function addVselectParity(spec, list, dest, cond, a, b, meta) {
  if (spec.idx_shifted) {
    addVselect(list, dest, cond, b, a, meta);
  } else {
    addVselect(list, dest, cond, a, b, meta);
  }
}

function selectionMode(spec, roundIdx) {
  const perRound = spec.selection_mode_by_round || {};
  if (roundIdx != null && perRound[roundIdx]) {
    return perRound[roundIdx];
  }
  if (spec.selection_mode) return spec.selection_mode;
  return spec.use_bitmask_selection ? 'bitmask' : 'eq';
}

function selectByEqAlu(spec, aluOps, flowOps, tmp, sel, idx, nodes, const_s, const_v, meta) {
  if (!nodes.length) {
    throw new Error('empty node list');
  }
  const baseAddr = nodes[0][1];
  let first = true;
  nodes.slice(1).forEach(([nodeIdx, nodeAddr]) => {
    addAluVec(aluOps, '==', sel, idx, idxConst(spec, const_s, nodeIdx), meta);
    if (first) {
      addVselect(flowOps, tmp, sel, nodeAddr, baseAddr, meta);
      first = false;
    } else {
      addVselect(flowOps, tmp, sel, nodeAddr, tmp, meta);
    }
  });
  return tmp;
}

function addVload(list, dest, addr, meta) {
  const newOp = new Op('load', ['vload', dest, addr], false, meta || null);
  recordOp(newOp);
  list.push(newOp);
}

function addLoadOffset(list, dest, addr, offset, meta) {
  const newOp = new Op('load', ['load_offset', dest, addr, offset], false, meta || null);
  recordOp(newOp);
  list.push(newOp);
}

function addVstore(list, addr, src, meta) {
  const newOp = new Op('store', ['vstore', addr, src], false, meta || null);
  recordOp(newOp);
  list.push(newOp);
}

function buildOps(spec, layout, orderedOps) {
  ORDERED_OPS = orderedOps || null;
  SEQ = 0;
  USE_VALU_SELECT = !!spec.valu_select;
  const valuOps = [];
  VALU_OPS_REF = valuOps;
  const aluOps = [];
  const flowOps = [];
  const loadOps = [];
  const storeOps = [];

  const { linear, bitwise } = splitHashStages();
  const selectionModesPerRound = Array.from({ length: spec.rounds }, (_, r) => selectionMode(spec, r));
  const useVectorMajor = selectionModesPerRound.some((mode) => ['bitmask', 'mask', 'mask_precompute'].includes(mode));
  const cachedRoundAliases = spec.cached_round_aliases || {};
  const cachedRoundDepths = spec.cached_round_depth || {};
  const cachedRoundX = spec.cached_round_x || {};
  const cachedRounds = new Set([
    ...spec.base_cached_rounds,
    ...Object.keys(cachedRoundAliases).map((v) => Number.parseInt(v, 10)),
    ...Object.keys(cachedRoundDepths).map((v) => Number.parseInt(v, 10)),
  ]);

  const depthFromRound = (r) => {
    if (r === 0 || r === 11) return 0;
    if (r === 1 || r === 12) return 1;
    if (r === 2 || r === 13) return 2;
    if (r === 3 || r === 14) return 3;
    return null;
  };

  const depthFromAlias = (val) => {
    if ([0, 11, 1, 12, 2, 13, 3, 14].includes(val)) return depthFromRound(val);
    if ([0, 1, 2, 3].includes(val)) return val;
    return null;
  };

  if (spec.include_setup) {
    Object.keys(layout.const_s)
      .map((v) => Number.parseInt(v, 10))
      .sort((a, b) => a - b)
      .forEach((val) => {
        if (val === 0 && spec.proof_skip_const_zero) return;
        addConst(loadOps, layout.const_s[val], val, { setup: true, const: val });
      });

    addLoad(loadOps, layout.forest_values_p, layout.const_s[4], { setup: true, ptr: 'forest_values_p' });
    addLoad(loadOps, layout.inp_indices_p, layout.const_s[5], { setup: true, ptr: 'inp_indices_p' });
    addLoad(loadOps, layout.inp_values_p, layout.const_s[6], { setup: true, ptr: 'inp_values_p' });

    const ptrEngine = spec.ptr_setup_engine || 'flow';
    if (ptrEngine === 'flow') {
      addFlowAddImm(flowOps, layout.idx_ptr[0], layout.inp_indices_p, 0, { setup: true });
      addFlowAddImm(flowOps, layout.val_ptr[0], layout.inp_values_p, 0, { setup: true });
      for (let v = 1; v < spec.vectors; v++) {
        addFlowAddImm(flowOps, layout.idx_ptr[v], layout.idx_ptr[v - 1], VLIW_VLEN, { setup: true });
        addFlowAddImm(flowOps, layout.val_ptr[v], layout.val_ptr[v - 1], VLIW_VLEN, { setup: true });
      }
    } else if (ptrEngine === 'alu') {
      const zero = layout.const_s[0];
      const vlenConst = layout.const_s[VLIW_VLEN];
      addAlu(aluOps, '+', layout.idx_ptr[0], layout.inp_indices_p, zero, { setup: true });
      addAlu(aluOps, '+', layout.val_ptr[0], layout.inp_values_p, zero, { setup: true });
      for (let v = 1; v < spec.vectors; v++) {
        addAlu(aluOps, '+', layout.idx_ptr[v], layout.idx_ptr[v - 1], vlenConst, { setup: true });
        addAlu(aluOps, '+', layout.val_ptr[v], layout.val_ptr[v - 1], vlenConst, { setup: true });
      }
    } else {
      throw new Error(`unknown ptr_setup_engine ${ptrEngine}`);
    }

    Object.keys(layout.const_v)
      .map((v) => Number.parseInt(v, 10))
      .sort((a, b) => a - b)
      .forEach((val) => {
        addVbroadcast(valuOps, layout.const_v[val], layout.const_s[val], { setup: true, const: val });
      });

    if (spec.node_ptr_incremental) {
      const zero = layout.const_s[0];
      const one = layout.const_s[1];
      const nodePtr = layout.inp_indices_p;
      addAlu(aluOps, '+', nodePtr, layout.forest_values_p, zero, { setup: true, node: 'base' });
      layout.node_v.forEach((vaddr, i) => {
        addLoad(loadOps, layout.node_tmp, nodePtr, { setup: true, node: i });
        addVbroadcast(valuOps, vaddr, layout.node_tmp, { setup: true, node: i });
        if (i + 1 < layout.node_v.length) {
          addAlu(aluOps, '+', nodePtr, nodePtr, one, { setup: true, node: 'inc' });
        }
      });
    } else {
      layout.node_v.forEach((vaddr, i) => {
        addAlu(aluOps, '+', layout.node_tmp, layout.forest_values_p, layout.const_s[i], { setup: true, node: i });
        addLoad(loadOps, layout.node_tmp, layout.node_tmp, { setup: true, node: i });
        addVbroadcast(valuOps, vaddr, layout.node_tmp, { setup: true, node: i });
      });
    }

    if (spec.idx_shifted) {
      addAlu(aluOps, '-', layout.node_tmp, layout.forest_values_p, layout.const_s[1], {
        setup: true,
        ptr: 'forest_values_p_shift',
      });
      addVbroadcast(valuOps, layout.forest_values_v, layout.node_tmp, {
        setup: true,
        ptr: 'forest_values_p_shift',
      });
    } else {
      addVbroadcast(valuOps, layout.forest_values_v, layout.forest_values_p, {
        setup: true,
        ptr: 'forest_values_p',
      });
    }

    for (let v = 0; v < spec.vectors; v++) {
      addVload(loadOps, layout.idx[v], layout.idx_ptr[v], { vec: v });
      if (spec.idx_shifted && !spec.proof_assume_shifted_input) {
        addValu(valuOps, '+', layout.idx[v], layout.idx[v], layout.const_v[1], {
          setup: true,
          vec: v,
          idx_shift: true,
        });
      }
      addVload(loadOps, layout.val[v], layout.val_ptr[v], { vec: v });
    }
  }

  let vecRoundPairs = [];
  const block = spec.vector_block || 0;
  if (block) {
    for (let blockStart = 0; blockStart < spec.vectors; blockStart += block) {
      const blockEnd = Math.min(spec.vectors, blockStart + block);
      for (let r = 0; r < spec.rounds; r++) {
        for (let v = blockStart; v < blockEnd; v++) {
          vecRoundPairs.push([v, r]);
        }
      }
    }
  } else if (useVectorMajor) {
    for (let v = 0; v < spec.vectors; v++) {
      for (let r = 0; r < spec.rounds; r++) {
        vecRoundPairs.push([v, r]);
      }
    }
  } else {
    for (let r = 0; r < spec.rounds; r++) {
      for (let v = 0; v < spec.vectors; v++) {
        vecRoundPairs.push([v, r]);
      }
    }
  }

  vecRoundPairs.forEach(([v, r]) => {
    const selectionModeRound = selectionModesPerRound[r];
    const maskMode = ['mask', 'mask_precompute'].includes(selectionModeRound);
    const maskPrecompute = selectionModeRound === 'mask_precompute' && layout.extra.length >= 4;
    const tmp = layout.tmp[v];
    const sel = layout.sel[v];
    let extra = null;
    let extra2 = null;
    let extra3 = null;
    let extraKey = null;
    let extra2Key = null;
    let extra3Key = null;
    const tmpReadKey = `tmp_read:${r}:${v}`;
    if (layout.extra.length) {
      extra = layout.extra[v % layout.extra.length];
      extraKey = `extra:${v % layout.extra.length}`;
      if (layout.extra.length > 1) {
        extra2 = layout.extra[(v + 1) % layout.extra.length];
        extra2Key = `extra:${(v + 1) % layout.extra.length}`;
      }
      if (layout.extra.length > 2) {
        extra3 = layout.extra[(v + 2) % layout.extra.length];
        extra3Key = `extra:${(v + 2) % layout.extra.length}`;
      }
    }
    const idx = layout.idx[v];
    const val = layout.val[v];
    const offloadNodeXor = !!spec.offload_node_xor;

    const nodeXor = (src, meta) => {
      addValu(valuOps, '^', val, val, src, meta, offloadNodeXor);
    };

    let bits0 = null;
    let bits1 = null;
    let data1 = null;
    let data2 = null;
    const rSelAlias = cachedRoundAliases[r] != null ? cachedRoundAliases[r] : r;

    let cacheDepth = null;
    if (cachedRoundDepths[r] != null) {
      cacheDepth = cachedRoundDepths[r];
    } else if (cachedRoundAliases[r] != null) {
      cacheDepth = depthFromAlias(cachedRoundAliases[r]);
    } else if (cachedRounds.has(r)) {
      cacheDepth = depthFromRound(r);
    }
    const cacheX = cacheDepth != null
      ? (cachedRoundX[r] != null ? cachedRoundX[r] : spec.vectors)
      : 0;

    if (maskPrecompute) {
      let maskDepth = null;
      if ([1, 2, 3].includes(cacheDepth) && v < cacheX) {
        maskDepth = cacheDepth;
      }
      if (spec.depth4_cached_rounds.includes(r) && v < spec.x4) {
        maskDepth = 4;
      }
      if (maskDepth != null) {
        [bits0, bits1, data1, data2] = layout.extra.slice(0, 4);
        const oneV = layout.const_v[1];
        addValu(valuOps, '&', bits0, idx, oneV, { round: r, vec: v, sel: 'mask_pre' });
        if (maskDepth >= 2) {
          addValu(valuOps, '>>', bits1, idx, oneV, { round: r, vec: v, sel: 'mask_pre' });
          addValu(valuOps, '&', bits1, bits1, oneV, { round: r, vec: v, sel: 'mask_pre' });
        }
      }
    }

    const rSel = [0, 1, 2, 3].includes(cacheDepth) ? cacheDepth : null;

    if (cacheDepth != null && v < cacheX) {
      if (rSel === 0) {
        nodeXor(layout.node_v[0], { round: r, vec: v });
      } else if (rSel === 1) {
        if (maskPrecompute && spec.idx_shifted) {
          addVselect(flowOps, sel, bits0, layout.node_v[2], layout.node_v[1], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          nodeXor(sel, { round: r, vec: v });
        } else if (selectionModeRound === 'mask' && spec.idx_shifted) {
          const oneV = layout.const_v[1];
          addValu(valuOps, '&', tmp, idx, oneV, { round: r, vec: v, sel: 'mask' });
          addVselect(flowOps, sel, tmp, layout.node_v[2], layout.node_v[1], tagTemp({ round: r, vec: v, sel: 'mask' }, tmpReadKey));
          nodeXor(sel, { round: r, vec: v });
        } else if (selectionModeRound === 'bitmask' && extra != null) {
          addAluVec(aluOps, '&', tmp, idx, layout.const_s[1], { round: r, vec: v });
          addVselectParity(spec, flowOps, sel, tmp, layout.node_v[1], layout.node_v[2], tagTemp({ round: r, vec: v }, tmpReadKey));
          nodeXor(sel, { round: r, vec: v });
        } else {
          const nodes = [
            [1, layout.node_v[1]],
            [2, layout.node_v[2]],
          ];
          selectByEqAlu(spec, aluOps, flowOps, tmp, sel, idx, nodes, layout.const_s, layout.const_v, { round: r, vec: v });
          nodeXor(tmp, { round: r, vec: v });
        }
      } else if (rSel === 2) {
        if (maskPrecompute && spec.idx_shifted) {
          addVselect(flowOps, sel, bits0, layout.node_v[4], layout.node_v[3], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data1, bits0, layout.node_v[6], layout.node_v[5], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, sel, bits1, data1, sel, tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          nodeXor(sel, { round: r, vec: v });
        } else if (selectionModeRound === 'mask' && spec.idx_shifted && extra != null) {
          const oneV = layout.const_v[1];
          const shift1 = layout.const_v[1];
          addValu(valuOps, '&', tmp, idx, oneV, { round: r, vec: v, sel: 'mask' });
          addVselect(flowOps, sel, tmp, layout.node_v[4], layout.node_v[3], tagTemp({ round: r, vec: v, sel: 'mask' }, tmpReadKey));
          addVselect(flowOps, extra, tmp, layout.node_v[6], layout.node_v[5], tagTemp(tagTemp({ round: r, vec: v, sel: 'mask' }, extraKey), tmpReadKey));
          addValu(valuOps, '>>', tmp, idx, shift1, { round: r, vec: v, sel: 'mask' });
          addValu(valuOps, '&', tmp, tmp, oneV, { round: r, vec: v, sel: 'mask' });
          addVselect(flowOps, sel, tmp, extra, sel, tagTemp({ round: r, vec: v, sel: 'mask' }, extraKey));
          nodeXor(sel, { round: r, vec: v });
        } else if (selectionModeRound === 'bitmask' && extra != null) {
          addAluVec(aluOps, '&', tmp, idx, layout.const_s[1], { round: r, vec: v });
          addVselectParity(spec, flowOps, sel, tmp, layout.node_v[3], layout.node_v[4], tagTemp({ round: r, vec: v }, tmpReadKey));
          addVselectParity(spec, flowOps, extra, tmp, layout.node_v[5], layout.node_v[6], tagTemp(tagTemp({ round: r, vec: v }, extraKey), tmpReadKey));
          addAluVec(aluOps, '<', tmp, idx, idxConst(spec, layout.const_s, 5), { round: r, vec: v });
          addVselect(flowOps, sel, tmp, sel, extra, tagTemp({ round: r, vec: v }, extraKey));
          nodeXor(sel, { round: r, vec: v });
        } else {
          const nodes = Array.from({ length: 4 }, (_, i) => [i + 3, layout.node_v[i + 3]]);
          selectByEqAlu(spec, aluOps, flowOps, tmp, sel, idx, nodes, layout.const_s, layout.const_v, { round: r, vec: v });
          nodeXor(tmp, { round: r, vec: v });
        }
      } else if (rSel === 3) {
        if (maskPrecompute && spec.idx_shifted) {
          const oneV = layout.const_v[1];
          addVselect(flowOps, sel, bits0, layout.node_v[8], layout.node_v[7], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data1, bits0, layout.node_v[10], layout.node_v[9], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, sel, bits1, data1, sel, tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data1, bits0, layout.node_v[12], layout.node_v[11], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data2, bits0, layout.node_v[14], layout.node_v[13], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data1, bits1, data2, data1, tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addValu(valuOps, '>>', tmp, idx, layout.const_v[2], { round: r, vec: v, sel: 'mask_pre' });
          addValu(valuOps, '&', tmp, tmp, oneV, { round: r, vec: v, sel: 'mask_pre' });
          addVselect(flowOps, sel, tmp, data1, sel, tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          nodeXor(sel, { round: r, vec: v });
        } else if (selectionModeRound === 'mask' && spec.idx_shifted && extra != null && extra2 != null) {
          const oneV = layout.const_v[1];
          const shift1 = layout.const_v[1];
          const shift2 = layout.const_v[2];
          addValu(valuOps, '&', tmp, idx, oneV, { round: r, vec: v, sel: 'mask' });
          addVselect(flowOps, sel, tmp, layout.node_v[8], layout.node_v[7], tagTemp({ round: r, vec: v, sel: 'mask' }, tmpReadKey));
          addVselect(flowOps, extra2, tmp, layout.node_v[10], layout.node_v[9], tagTemp(tagTemp({ round: r, vec: v, sel: 'mask' }, extra2Key), tmpReadKey));
          addValu(valuOps, '>>', tmp, idx, shift1, { round: r, vec: v, sel: 'mask' });
          addValu(valuOps, '&', tmp, tmp, oneV, { round: r, vec: v, sel: 'mask' });
          addVselect(flowOps, sel, tmp, extra2, sel, tagTemp({ round: r, vec: v, sel: 'mask' }, extra2Key));
          addValu(valuOps, '&', tmp, idx, oneV, { round: r, vec: v, sel: 'mask' });
          addVselect(flowOps, extra, tmp, layout.node_v[12], layout.node_v[11], tagTemp(tagTemp({ round: r, vec: v, sel: 'mask' }, extraKey), tmpReadKey));
          addVselect(flowOps, extra2, tmp, layout.node_v[14], layout.node_v[13], tagTemp(tagTemp({ round: r, vec: v, sel: 'mask' }, extra2Key), tmpReadKey));
          addValu(valuOps, '>>', tmp, idx, shift1, { round: r, vec: v, sel: 'mask' });
          addValu(valuOps, '&', tmp, tmp, oneV, { round: r, vec: v, sel: 'mask' });
          addVselect(flowOps, extra, tmp, extra2, extra, tagTemp({ round: r, vec: v, sel: 'mask' }, extra2Key || extraKey));
          addValu(valuOps, '>>', tmp, idx, shift2, { round: r, vec: v, sel: 'mask' });
          addValu(valuOps, '&', tmp, tmp, oneV, { round: r, vec: v, sel: 'mask' });
          addVselect(flowOps, sel, tmp, extra, sel, tagTemp({ round: r, vec: v, sel: 'mask' }, extraKey));
          nodeXor(sel, { round: r, vec: v });
        } else if (selectionModeRound === 'bitmask' && extra != null) {
          addAluVec(aluOps, '&', tmp, idx, layout.const_s[1], { round: r, vec: v });
          addVselectParity(spec, flowOps, sel, tmp, layout.node_v[7], layout.node_v[8], tagTemp({ round: r, vec: v }, tmpReadKey));
          if (extra2 != null) {
            addVselectParity(spec, flowOps, extra2, tmp, layout.node_v[9], layout.node_v[10], tagTemp(tagTemp({ round: r, vec: v }, extra2Key), tmpReadKey));
            addAluVec(aluOps, '<', tmp, idx, idxConst(spec, layout.const_s, 9), { round: r, vec: v });
            addVselect(flowOps, sel, tmp, sel, extra2, tagTemp({ round: r, vec: v }, extra2Key));
          } else {
            addVselect(flowOps, extra, tmp, layout.node_v[9], layout.node_v[10], tagTemp(tagTemp({ round: r, vec: v }, extraKey), tmpReadKey));
            addAluVec(aluOps, '<', tmp, idx, idxConst(spec, layout.const_s, 9), { round: r, vec: v });
            addVselect(flowOps, sel, tmp, sel, extra, tagTemp({ round: r, vec: v }, extraKey));
          }
          addAluVec(aluOps, '&', tmp, idx, layout.const_s[1], { round: r, vec: v });
          addVselectParity(spec, flowOps, extra, tmp, layout.node_v[11], layout.node_v[12], tagTemp(tagTemp({ round: r, vec: v }, extraKey), tmpReadKey));
          addVselectParity(spec, flowOps, extra2 != null ? extra2 : sel, tmp, layout.node_v[13], layout.node_v[14], tagTemp(tagTemp({ round: r, vec: v }, extra2Key || extraKey), tmpReadKey));
          addAluVec(aluOps, '<', tmp, idx, idxConst(spec, layout.const_s, 13), { round: r, vec: v });
          addVselect(flowOps, sel, tmp, extra, sel, tagTemp({ round: r, vec: v }, extraKey));
          nodeXor(sel, { round: r, vec: v });
        } else {
          const nodes = Array.from({ length: 8 }, (_, i) => [i + 7, layout.node_v[i + 7]]);
          selectByEqAlu(spec, aluOps, flowOps, tmp, sel, idx, nodes, layout.const_s, layout.const_v, { round: r, vec: v });
          nodeXor(tmp, { round: r, vec: v });
        }
      } else if (cacheDepth === 4 && v < spec.x4) {
        if (maskPrecompute && spec.idx_shifted) {
          const oneV = layout.const_v[1];
          const shift1 = layout.const_v[1];
          const shift2 = layout.const_v[2];
          const shift3 = layout.const_v[3];
          [bits0, bits1, data1, data2] = layout.extra.slice(0, 4);
          addVselect(flowOps, sel, bits0, layout.node_v[16], layout.node_v[15], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data1, bits0, layout.node_v[18], layout.node_v[17], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, sel, bits1, data1, sel, tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data1, bits0, layout.node_v[20], layout.node_v[19], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data2, bits0, layout.node_v[22], layout.node_v[21], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data1, bits1, data2, data1, tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addValu(valuOps, '>>', tmp, idx, shift2, { round: r, vec: v, sel: 'mask_pre' });
          addValu(valuOps, '&', tmp, tmp, oneV, { round: r, vec: v, sel: 'mask_pre' });
          addVselect(flowOps, sel, tmp, data1, sel, tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data1, bits0, layout.node_v[24], layout.node_v[23], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data2, bits0, layout.node_v[26], layout.node_v[25], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data1, bits1, data2, data1, tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data2, bits0, layout.node_v[28], layout.node_v[27], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, extra, bits0, layout.node_v[30], layout.node_v[29], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data2, bits1, extra, data2, tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addValu(valuOps, '>>', tmp, idx, shift2, { round: r, vec: v, sel: 'mask_pre' });
          addValu(valuOps, '&', tmp, tmp, oneV, { round: r, vec: v, sel: 'mask_pre' });
          addVselect(flowOps, data1, tmp, data2, data1, tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addValu(valuOps, '>>', tmp, idx, shift3, { round: r, vec: v, sel: 'mask_pre' });
          addValu(valuOps, '&', tmp, tmp, oneV, { round: r, vec: v, sel: 'mask_pre' });
          addVselect(flowOps, sel, tmp, data1, sel, tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          nodeXor(sel, { round: r, vec: v });
        } else {
          const nodes = Array.from({ length: 16 }, (_, i) => [i + 15, layout.node_v[i + 15]]);
          selectByEqAlu(spec, aluOps, flowOps, tmp, sel, idx, nodes, layout.const_s, layout.const_v, { round: r, vec: v });
          nodeXor(tmp, { round: r, vec: v });
        }
      }
    } else {
      if (selectionModeRound === 'eq') {
        addAluVec(aluOps, '==', tmp, idx, layout.const_s[0], { round: r, vec: v });
        addVselect(flowOps, sel, tmp, layout.node_v[0], layout.node_v[1], tagTemp({ round: r, vec: v }, tmpReadKey));
        nodeXor(sel, { round: r, vec: v });
      } else if (selectionModeRound === 'bitmask' && extra != null) {
        addAluVec(aluOps, '&', tmp, idx, layout.const_s[1], { round: r, vec: v });
        addVselectParity(spec, flowOps, sel, tmp, layout.node_v[0], layout.node_v[1], tagTemp({ round: r, vec: v }, tmpReadKey));
        nodeXor(sel, { round: r, vec: v });
      } else if (selectionModeRound === 'mask' && extra != null && spec.idx_shifted) {
        const oneV = layout.const_v[1];
        addValu(valuOps, '&', tmp, idx, oneV, { round: r, vec: v, sel: 'mask' });
        addVselect(flowOps, sel, tmp, layout.node_v[1], layout.node_v[2], tagTemp({ round: r, vec: v, sel: 'mask' }, tmpReadKey));
        nodeXor(sel, { round: r, vec: v });
      }
    }

    if (spec.reset_on_valu) {
      if (spec.offload_hash_op1 && linear.length) {
        const { mult, add } = linear[0];
        addVmuladd(valuOps, val, val, layout.const_v[mult], layout.const_v[add], { round: r, vec: v, hash: 0 });
      } else if (linear.length) {
        const { mult, add } = linear[0];
        addVmuladd(valuOps, val, val, layout.const_v[mult], layout.const_v[add], { round: r, vec: v, hash: 0 });
      }
      for (let i = 1; i < linear.length; i++) {
        const stage = linear[i];
        addVmuladd(valuOps, val, val, layout.const_v[stage.mult], layout.const_v[stage.add], { round: r, vec: v, hash: i });
      }
    } else {
      const shiftOnValu = spec.shifts_on_valu;
      const tmp2 = layout.tmp2[v];
      linear.forEach((stage, i) => {
        addVmuladd(valuOps, val, val, layout.const_v[stage.mult], layout.const_v[stage.add], { round: r, vec: v, hash: i });
      });
      bitwise.forEach((stage, i) => {
        if (stage.op2 === '^') {
          addValu(valuOps, '^', val, val, layout.const_v[stage.const], { round: r, vec: v, hash: i });
        } else {
          addValu(valuOps, '+', val, val, layout.const_v[stage.const], { round: r, vec: v, hash: i });
        }
        if (shiftOnValu) {
          addValu(valuOps, stage.shift_op, tmp2, val, layout.const_v[stage.shift], { round: r, vec: v, hash: i });
          addValu(valuOps, stage.op2, val, val, tmp2, { round: r, vec: v, hash: i });
        } else {
          addValu(valuOps, stage.shift_op, tmp2, val, layout.const_v[stage.shift], { round: r, vec: v, hash: i });
          addAluVec(aluOps, stage.op2, val, val, tmp2, { round: r, vec: v, hash: i });
        }
      });
    }

    if (spec.selection_mode === 'mask_precompute' && extra3 != null && maskMode) {
      const maskDepth = cacheDepth;
      if (maskDepth != null && v < cacheX) {
        addValu(valuOps, '&', extra3, idx, layout.const_v[1], { round: r, vec: v, sel: 'mask_pre' });
      }
    }

    if (spec.selection_mode === 'mask_precompute' && maskMode && selectionModeRound === 'mask_precompute') {
      const maskDepth = cacheDepth;
      if (maskDepth != null && v < cacheX) {
        const countIndex = idxConst(spec, layout.const_s, 1);
        addAluVec(aluOps, '<', tmp, idx, countIndex, { round: r, vec: v, sel: 'mask_pre' });
      }
    }

    const nodeXorFlow = spec.offload_node_xor;
    if (!nodeXorFlow && selectionModeRound !== 'mask_precompute') {
      addAluVec(aluOps, '<', tmp, val, layout.const_s[1], { round: r, vec: v, parity: true });
    } else if (selectionModeRound !== 'mask_precompute') {
      addValu(valuOps, '<', tmp, val, layout.const_v[1], { round: r, vec: v, parity: true });
    }

    if (selectionModeRound === 'bitmask' && extra != null) {
      addAluVec(aluOps, '<', tmp, idx, idxConst(spec, layout.const_s, 7), { round: r, vec: v });
      addVselect(flowOps, idx, tmp, layout.const_v[1], layout.const_v[2], tagTemp({ round: r, vec: v }, tmpReadKey));
    } else {
      addAluVec(aluOps, '+', idx, idx, idxConst(spec, layout.const_s, 1), { round: r, vec: v });
    }

    addLoadOffset(loadOps, tmp, layout.forest_values_p, idx, { round: r, vec: v });
    if (spec.offload_node_xor) {
      addValu(valuOps, '^', val, val, tmp, { round: r, vec: v });
    } else {
      addAluVec(aluOps, '^', val, val, tmp, { round: r, vec: v });
    }

    if (spec.selection_mode === 'mask_precompute' && maskMode && selectionModeRound === 'mask_precompute') {
      addValu(valuOps, '&', tmp, idx, layout.const_v[1], { round: r, vec: v, sel: 'mask_pre' });
      addVselect(flowOps, idx, tmp, layout.const_v[1], layout.const_v[2], tagTemp({ round: r, vec: v }, tmpReadKey));
    } else {
      addAluVec(aluOps, '<', tmp, val, layout.const_s[1], { round: r, vec: v });
      addVselect(flowOps, idx, tmp, layout.const_v[1], layout.const_v[2], tagTemp({ round: r, vec: v }, tmpReadKey));
    }
  });

  for (let v = 0; v < spec.vectors; v++) {
    addVstore(storeOps, layout.val_ptr[v], layout.val[v], { vec: v });
    addVstore(storeOps, layout.idx_ptr[v], layout.idx[v], { vec: v });
  }

  return {
    valu_ops: valuOps,
    alu_ops: aluOps,
    flow_ops: flowOps,
    load_ops: loadOps,
    store_ops: storeOps,
  };
}

function buildSetupPrelude(spec, layout, caps) {
  const setupInstrs = [];
  const pack = (engine, slots) => {
    const cap = caps?.[engine] ?? VLIW_SLOT_LIMITS[engine] ?? 0;
    if (cap <= 0) return;
    for (let i = 0; i < slots.length; i += cap) {
      setupInstrs.push({ [engine]: slots.slice(i, i + cap) });
    }
  };

  const constLoads = [];
  Object.keys(layout.const_s)
    .map((v) => Number.parseInt(v, 10))
    .sort((a, b) => a - b)
    .forEach((val) => {
      if (val === 0 && spec.proof_skip_const_zero) return;
      constLoads.push(['const', layout.const_s[val], val]);
    });
  pack('load', constLoads);

  const ptrLoads = [
    ['load', layout.forest_values_p, layout.const_s[4]],
    ['load', layout.inp_indices_p, layout.const_s[5]],
    ['load', layout.inp_values_p, layout.const_s[6]],
  ];
  pack('load', ptrLoads);

  if (spec.idx_shifted) {
    setupInstrs.push({ alu: [['-', layout.node_tmp, layout.forest_values_p, layout.const_s[1]]] });
    setupInstrs.push({ valu: [['vbroadcast', layout.forest_values_v, layout.node_tmp]] });
  } else {
    setupInstrs.push({ valu: [['vbroadcast', layout.forest_values_v, layout.forest_values_p]] });
  }

  const ptrEngine = spec.ptr_setup_engine || 'flow';
  if (ptrEngine === 'flow') {
    const flowSetup = [
      ['add_imm', layout.idx_ptr[0], layout.inp_indices_p, 0],
      ['add_imm', layout.val_ptr[0], layout.inp_values_p, 0],
    ];
    for (let v = 1; v < spec.vectors; v++) {
      flowSetup.push(['add_imm', layout.idx_ptr[v], layout.idx_ptr[v - 1], VLIW_VLEN]);
      flowSetup.push(['add_imm', layout.val_ptr[v], layout.val_ptr[v - 1], VLIW_VLEN]);
    }
    pack('flow', flowSetup);
  } else if (ptrEngine === 'alu') {
    const zero = layout.const_s[0];
    const vlenConst = layout.const_s[VLIW_VLEN];
    setupInstrs.push({ alu: [['+', layout.idx_ptr[0], layout.inp_indices_p, zero]] });
    setupInstrs.push({ alu: [['+', layout.val_ptr[0], layout.inp_values_p, zero]] });
    for (let v = 1; v < spec.vectors; v++) {
      setupInstrs.push({ alu: [['+', layout.idx_ptr[v], layout.idx_ptr[v - 1], vlenConst]] });
      setupInstrs.push({ alu: [['+', layout.val_ptr[v], layout.val_ptr[v - 1], vlenConst]] });
    }
  } else {
    throw new Error(`unknown ptr_setup_engine ${ptrEngine}`);
  }

  if (spec.node_ptr_incremental) {
    const zero = layout.const_s[0];
    const one = layout.const_s[1];
    let nodePtr = layout.inp_indices_p;
    setupInstrs.push({ alu: [['+', nodePtr, layout.forest_values_p, zero]] });
    for (let i = 0; i < layout.node_v.length; i++) {
      setupInstrs.push({ load: [['load', layout.node_tmp, nodePtr]] });
      setupInstrs.push({ valu: [['vbroadcast', layout.node_v[i], layout.node_tmp]] });
      if (i + 1 < layout.node_v.length) {
        setupInstrs.push({ alu: [['+', nodePtr, nodePtr, one]] });
      }
    }
  } else {
    for (let i = 0; i < layout.node_v.length; i++) {
      setupInstrs.push({ alu: [['+', layout.node_tmp, layout.forest_values_p, layout.const_s[i]]] });
      setupInstrs.push({ load: [['load', layout.node_tmp, layout.node_tmp]] });
      setupInstrs.push({ valu: [['vbroadcast', layout.node_v[i], layout.node_tmp]] });
    }
  }

  const constVBroadcasts = Object.keys(layout.const_v)
    .map((v) => Number.parseInt(v, 10))
    .sort((a, b) => a - b)
    .map((val) => ['vbroadcast', layout.const_v[val], layout.const_s[val]]);
  pack('valu', constVBroadcasts);

  const vloads = [];
  for (let v = 0; v < spec.vectors; v++) {
    vloads.push(['vload', layout.idx[v], layout.idx_ptr[v]]);
    vloads.push(['vload', layout.val[v], layout.val_ptr[v]]);
  }
  pack('load', vloads);

  if (spec.idx_shifted && !spec.proof_assume_shifted_input) {
    const shiftOps = Array.from({ length: spec.vectors }, (_, v) => ['+', layout.idx[v], layout.idx[v], layout.const_v[1]]);
    pack('valu', shiftOps);
  }

  return setupInstrs;
}

function buildFinalOps(spec) {
  const layout = buildLayout(spec);
  let setupOps = [];
  if (spec.setup_style === 'packed') {
    const setupInstrs = buildSetupPrelude(spec, layout, resolveCaps(spec));
    setupInstrs.forEach((instr) => {
      Object.entries(instr).forEach(([eng, slots]) => {
        slots.forEach((slot) => {
          setupOps.push(new Op(eng, slot, false, { setup: true }));
        });
      });
    });
  }

  let specForOps = spec;
  if ((spec.setup_style === 'packed' || spec.setup_style === 'none') && spec.include_setup) {
    specForOps = { ...spec, include_setup: false };
  }

  const orderedOps = [];
  buildOps(specForOps, layout, orderedOps);
  const offloadableCount = orderedOps.reduce((count, op) => (op.offloadable ? count + 1 : count), 0);

  let finalOps = [];
  let offloaded = 0;
  setupOps.concat(orderedOps).forEach((op) => {
    if (op.offloadable && offloaded < spec.offload_op1) {
      const [opName, dest, a, b] = op.slot;
      for (let lane = 0; lane < VLIW_VLEN; lane++) {
        finalOps.push(new Op('alu', [opName, dest + lane, a + lane, b + lane], false, op.meta));
      }
      offloaded += 1;
    } else {
      finalOps.push(op);
    }
  });

  const padCycles = spec.valu_pad_cycles || 0;
  if (padCycles) {
    const padCount = padCycles * spec.valu_cap;
    const padDest = layout.tmp[0];
    for (let i = 0; i < padCount; i++) {
      finalOps.unshift(new Op('valu', ['^', padDest, padDest, padDest]));
    }
  }

  finalOps.forEach((op, idx) => {
    op.id = idx;
  });

  return { ops: finalOps, offloadableCount };
}

function vecAddrs(base) {
  return Array.from({ length: VLIW_VLEN }, (_, i) => base + i);
}

function readsWrites(op) {
  const { engine, slot } = op;
  if (engine === 'alu') {
    const [, dest, a1, a2] = slot;
    return { reads: [a1, a2], writes: [dest] };
  }
  if (engine === 'load') {
    switch (slot[0]) {
      case 'const':
        return { reads: [], writes: [slot[1]] };
      case 'load':
        return { reads: [slot[2]], writes: [slot[1]] };
      case 'vload':
        return { reads: [slot[2]], writes: vecAddrs(slot[1]) };
      case 'load_offset':
        return {
          reads: [slot[2], ...vecAddrs(slot[3])],
          writes: vecAddrs(slot[1]),
        };
      default:
        throw new Error(`Unknown load op ${slot[0]}`);
    }
  }
  if (engine === 'store') {
    switch (slot[0]) {
      case 'store':
        return { reads: [slot[1], slot[2]], writes: [] };
      case 'vstore':
        return { reads: [slot[1], ...vecAddrs(slot[2])], writes: [] };
      default:
        throw new Error(`Unknown store op ${slot[0]}`);
    }
  }
  if (engine === 'flow') {
    switch (slot[0]) {
      case 'add_imm':
        return { reads: [slot[2]], writes: [slot[1]] };
      case 'select':
        return { reads: [slot[2], slot[3], slot[4]], writes: [slot[1]] };
      case 'vselect':
        return {
          reads: [...vecAddrs(slot[2]), ...vecAddrs(slot[3]), ...vecAddrs(slot[4])],
          writes: vecAddrs(slot[1]),
        };
      default:
        throw new Error(`Unknown flow op ${slot[0]}`);
    }
  }
  if (engine === 'valu') {
    switch (slot[0]) {
      case 'vbroadcast':
        return { reads: [slot[2]], writes: vecAddrs(slot[1]) };
      case 'multiply_add':
        return {
          reads: [...vecAddrs(slot[2]), ...vecAddrs(slot[3]), ...vecAddrs(slot[4])],
          writes: vecAddrs(slot[1]),
        };
      default:
        return { reads: [...vecAddrs(slot[2]), ...vecAddrs(slot[3])], writes: vecAddrs(slot[1]) };
    }
  }
  if (engine === 'debug') {
    return { reads: [], writes: [] };
  }
  throw new Error(`Unknown engine ${engine}`);
}

function buildDeps(ops) {
  const readsList = [];
  const writesList = [];
  ops.forEach((op) => {
    const { reads, writes } = readsWrites(op);
    readsList.push(reads);
    writesList.push(writes);
  });

  const deps = Array.from({ length: ops.length }, () => []);
  const lastWrite = new Map();
  const lastRead = new Map();
  const lastTemp = new Map();

  for (let i = 0; i < ops.length; i++) {
    const reads = readsList[i];
    const writes = writesList[i];
    reads.forEach((addr) => {
      if (lastWrite.has(addr)) deps[i].push(lastWrite.get(addr));
    });
    writes.forEach((addr) => {
      if (lastWrite.has(addr)) deps[i].push(lastWrite.get(addr));
      if (lastRead.has(addr)) deps[i].push(lastRead.get(addr));
    });
    let temps = [];
    const tempMeta = ops[i].meta?.temp;
    if (tempMeta) {
      temps = typeof tempMeta === 'string' ? [tempMeta] : Array.from(tempMeta);
    }
    temps.forEach((key) => {
      if (lastTemp.has(key)) deps[i].push(lastTemp.get(key));
    });

    reads.forEach((addr) => lastRead.set(addr, i));
    writes.forEach((addr) => {
      lastWrite.set(addr, i);
      lastRead.delete(addr);
    });
    temps.forEach((key) => lastTemp.set(key, i));
  }

  return { deps, readsList, writesList };
}

class MinHeap {
  constructor(compare) {
    this.compare = compare;
    this.data = [];
  }

  push(item) {
    const data = this.data;
    data.push(item);
    let idx = data.length - 1;
    while (idx > 0) {
      const parent = Math.floor((idx - 1) / 2);
      if (this.compare(data[idx], data[parent]) >= 0) break;
      [data[idx], data[parent]] = [data[parent], data[idx]];
      idx = parent;
    }
  }

  pop() {
    const data = this.data;
    if (!data.length) return null;
    const root = data[0];
    const tail = data.pop();
    if (data.length && tail) {
      data[0] = tail;
      let idx = 0;
      while (true) {
        const left = idx * 2 + 1;
        const right = left + 1;
        let smallest = idx;
        if (left < data.length && this.compare(data[left], data[smallest]) < 0) {
          smallest = left;
        }
        if (right < data.length && this.compare(data[right], data[smallest]) < 0) {
          smallest = right;
        }
        if (smallest === idx) break;
        [data[idx], data[smallest]] = [data[smallest], data[idx]];
        idx = smallest;
      }
    }
    return root;
  }

  peek() {
    return this.data.length ? this.data[0] : null;
  }

  get size() {
    return this.data.length;
  }
}

function scheduleOpsDepOnce(ops, caps, { returnOps = false, seed = 0, jitter = 0 } = {}) {
  const { deps, readsList, writesList } = buildDeps(ops);
  const nOps = ops.length;
  const children = Array.from({ length: nOps }, () => []);
  const indegree = Array.from({ length: nOps }, () => 0);

  for (let i = 0; i < nOps; i++) {
    deps[i].forEach((d) => {
      children[d].push([i, 1]);
      indegree[i] += 1;
    });
  }

  const priority = Array.from({ length: nOps }, () => 1);
  for (let i = nOps - 1; i >= 0; i--) {
    if (children[i].length) {
      let maxChild = 0;
      children[i].forEach(([c]) => {
        if (priority[c] > maxChild) maxChild = priority[c];
      });
      priority[i] = 1 + maxChild;
    }
  }

  const earliest = Array.from({ length: nOps }, () => 0);
  const scheduled = Array.from({ length: nOps }, () => -1);
  const ready = {};
  const rng = jitter > 0 ? mulberry32(seed) : null;

  const jitterKey = () => (rng ? rng() * jitter : 0);

  for (let i = 0; i < nOps; i++) {
    if (indegree[i] === 0) {
      const engine = ops[i].engine;
      if (!ready[engine]) {
        ready[engine] = new MinHeap(compareTuple);
      }
      ready[engine].push([0, -priority[i], jitterKey(), i]);
    }
  }

  const engineOrderBase = ['valu', 'alu', 'flow', 'load', 'store', 'debug'];
  const engineIndex = {};
  engineOrderBase.forEach((eng, idx) => {
    engineIndex[eng] = idx;
  });

  const instrs = [];
  let cycle = 0;
  let remaining = nOps;
  while (remaining > 0) {
    while (instrs.length <= cycle) {
      instrs.push({});
    }

    const writesCycle = new Set();
    const engineCounts = {};
    let anyScheduled = false;

    const releaseChildren = (idx) => {
      children[idx].forEach(([child, latency]) => {
        indegree[child] -= 1;
        earliest[child] = Math.max(earliest[child], scheduled[idx] + latency);
        if (indegree[child] === 0) {
          const childEngine = ops[child].engine;
          if (!ready[childEngine]) {
            ready[childEngine] = new MinHeap(compareTuple);
          }
          ready[childEngine].push([earliest[child], -priority[child], jitterKey(), child]);
        }
      });
    };

    let madeProgress = true;
    while (madeProgress) {
      madeProgress = false;
      const engineKey = (engine) => {
        const cap = caps?.[engine] ?? 0;
        if (cap <= 0 || (engineCounts[engine] || 0) >= cap) {
          return [0, -1, -engineIndex[engine]];
        }
        const heap = ready[engine];
        if (!heap || heap.size === 0) {
          return [0, -1, -engineIndex[engine]];
        }
        const top = heap.peek();
        if (top[0] > cycle) {
          return [0, -1, -engineIndex[engine]];
        }
        return [1, -top[1], -engineIndex[engine]];
      };

      const engineOrder = engineOrderBase.slice().sort((a, b) => {
        const ka = engineKey(a);
        const kb = engineKey(b);
        return compareTuple(kb, ka);
      });

      engineOrder.forEach((engine) => {
        const cap = caps?.[engine] ?? 0;
        if (cap <= 0) return;
        let count = engineCounts[engine] || 0;
        if (count >= cap) return;
        const heap = ready[engine];
        if (!heap || heap.size === 0) return;
        const skipped = [];
        while (heap.size && count < cap) {
          const [readyCycle, negPri, j, idx] = heap.pop();
          if (readyCycle > cycle) {
            skipped.push([readyCycle, negPri, j, idx]);
            break;
          }
          const writes = writesList[idx];
          if (writes.some((w) => writesCycle.has(w))) {
            skipped.push([readyCycle, negPri, j, idx]);
            continue;
          }
          const op = ops[idx];
          if (!instrs[cycle][engine]) instrs[cycle][engine] = [];
          instrs[cycle][engine].push(returnOps ? op : op.slot);
          scheduled[idx] = cycle;
          writes.forEach((w) => writesCycle.add(w));
          remaining -= 1;
          anyScheduled = true;
          madeProgress = true;
          count += 1;
          releaseChildren(idx);
        }
        skipped.forEach((item) => heap.push(item));
        engineCounts[engine] = count;
      });
    }

    if (!anyScheduled) {
      let nextCycle = null;
      Object.values(ready).forEach((heap) => {
        if (heap.size) {
          const rc = heap.peek()[0];
          if (nextCycle == null || rc < nextCycle) {
            nextCycle = rc;
          }
        }
      });
      if (nextCycle == null) break;
      cycle = Math.max(cycle + 1, nextCycle);
      continue;
    }

    cycle += 1;
  }

  return instrs;
}

function scheduleOpsDep(ops, caps, { returnOps = false, seed = 0, jitter = 0, restarts = 1 } = {}) {
  const safeRestarts = Math.max(1, Math.floor(restarts));
  if (safeRestarts === 1 || jitter <= 0) {
    return scheduleOpsDepOnce(ops, caps, { returnOps, seed, jitter });
  }
  let bestInstrs = null;
  let bestCycles = null;
  for (let k = 0; k < safeRestarts; k++) {
    const instrs = scheduleOpsDepOnce(ops, caps, { returnOps, seed: seed + k, jitter });
    const cycles = countCycles(instrs);
    if (bestCycles == null || cycles < bestCycles) {
      bestCycles = cycles;
      bestInstrs = instrs;
    }
  }
  return bestInstrs || [];
}

function countCycles(instrs) {
  let cycles = 0;
  instrs.forEach((bundle) => {
    for (const [engine, slots] of Object.entries(bundle)) {
      if (engine !== 'debug' && slots && slots.length) {
        cycles += 1;
        break;
      }
    }
  });
  return cycles;
}

function compareTuple(a, b) {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  if (a.length < b.length) return -1;
  if (a.length > b.length) return 1;
  return 0;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function buildVliwDatasetFromSpec(specInput) {
  const spec = normalizeSpec(specInput);
  const caps = resolveCaps(spec);
  const { ops, offloadableCount } = buildFinalOps(spec);
  const { deps, readsList, writesList } = buildDeps(ops);
  const tasks = ops.map((op, idx) => ({
    id: idx,
    engine: op.engine,
    reads: readsList[idx],
    writes: writesList[idx],
    deps: deps[idx],
    bundle: null,
  }));

  const baselineInstrs = scheduleOpsDep(ops, caps, {
    returnOps: true,
    seed: spec.sched_seed ?? 0,
    jitter: spec.sched_jitter ?? 0,
    restarts: spec.sched_restarts ?? 1,
  });
  baselineInstrs.forEach((bundle, cycle) => {
    Object.values(bundle).forEach((slots) => {
      slots.forEach((op) => {
        const task = tasks[op.id];
        if (task) task.bundle = cycle;
      });
    });
  });

  const bundleCount = baselineInstrs.length;
  const baselineCycles = countCycles(baselineInstrs);

  return {
    version: 1,
    label: 'VLIW SIMD schedule (generated)',
    source: 'generated-in-browser',
    spec,
    tasks,
    taskCount: tasks.length,
    bundleCount,
    baselineCycles,
    caps,
    dag: {
      taskCount: tasks.length,
      caps,
      hash: null,
    },
    dependencyModel: { ...DEFAULT_DEPENDENCY_MODEL },
    offloadableCount,
  };
}

export function getDefaultSpec() {
  return normalizeSpec({});
}
