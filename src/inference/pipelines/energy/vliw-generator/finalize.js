import { VLEN as VLIW_VLEN } from '../vliw-shared.js';
import { Op, buildLayout } from './layout.js';
import { buildOps } from './ops.js';
import { buildSetupPrelude } from './setup.js';
import { resolveCaps } from './spec.js';

export function buildFinalOps(spec) {
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
