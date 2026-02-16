import { VLEN as VLIW_VLEN } from '../vliw-shared.js';
import { Op } from './layout.js';
import { splitHashStages } from './hash.js';

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

export function buildOps(spec, layout, orderedOps) {
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
          addVselect(flowOps, data1, bits0, layout.node_v[16], layout.node_v[15], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
          addVselect(flowOps, data2, bits0, layout.node_v[18], layout.node_v[17], tagTemp({ round: r, vec: v, sel: 'mask_pre' }, tmpReadKey));
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
      const offloadHashShift = !!spec.offload_hash_shift;
      const offloadHashOp1 = spec.offload_hash_op1 !== false;
      const offloadHashOp2 = !!spec.offload_hash_op2;
      linear.forEach((stage, i) => {
        addVmuladd(valuOps, val, val, layout.const_v[stage.mult], layout.const_v[stage.add], { round: r, vec: v, hash: i });
      });
      bitwise.forEach((stage, i) => {
        addValu(
          valuOps,
          stage.shift_op,
          tmp2,
          val,
          layout.const_v[stage.shift],
          { round: r, vec: v, hash: i },
          offloadHashShift,
        );
        addValu(
          valuOps,
          stage.op1,
          val,
          val,
          layout.const_v[stage.const],
          { round: r, vec: v, hash: i },
          offloadHashOp1,
        );
        if (shiftOnValu) {
          addValu(
            valuOps,
            stage.op2,
            val,
            val,
            tmp2,
            { round: r, vec: v, hash: i },
            offloadHashOp2,
          );
        } else {
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
  });

  if (spec.include_setup) {
    const constant = layout.const_s[spec.flow_setup];
    for (let v = 0; v < spec.vectors; v++) {
      addFlowAddImm(flowOps, layout.idx_ptr[v], layout.idx_ptr[v], spec.flow_setup, { setup: true });
      addFlowAddImm(flowOps, layout.val_ptr[v], layout.val_ptr[v], spec.flow_setup, { setup: true });
      if (spec.idx_shifted) {
        addFlowAddImm(flowOps, layout.idx_ptr[v], layout.idx_ptr[v], 1, { setup: true });
      }
      addLoad(loadOps, layout.idx_ptr[v], layout.idx_ptr[v], { setup: true });
      addLoad(loadOps, layout.val_ptr[v], layout.val_ptr[v], { setup: true });
    }
    if (constant) {
      addAlu(aluOps, '+', layout.forest_values_p, layout.forest_values_p, constant, { setup: true });
    }
  }

  return {
    valu_ops: valuOps,
    alu_ops: aluOps,
    flow_ops: flowOps,
    load_ops: loadOps,
    store_ops: storeOps,
  };
}
