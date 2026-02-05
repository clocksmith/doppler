import { VLEN as VLIW_VLEN } from '../vliw-shared.js';
import { HASH_STAGES } from './hash.js';

export class Op {
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

export function buildLayout(spec) {
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
