import { SLOT_LIMITS as VLIW_SLOT_LIMITS, VLEN as VLIW_VLEN } from '../vliw-shared.js';

export function buildSetupPrelude(spec, layout, caps) {
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
