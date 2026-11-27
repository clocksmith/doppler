// Splits each compiled step between two evaluators at a layer boundary (the homomorphic split). Stage A runs the
// step's operations before the cut and holds the encrypted KV cache; stage B runs the rest and returns the result.
// What crosses from A to B is exactly the set of ciphertext registers that B reads but does not define: the carry.
// Nothing is recompiled: both stages run the same compiled program, each on its own operation range, so the split
// output is the whole program's output bit for bit.

export const SPLIT_LAYER = 'ln2';
export const SPLIT_ROLES = ['stage-a', 'stage-b'];

export function splitStep(program, step, cutLayer = SPLIT_LAYER) {
  const layer = (step.layers ?? []).find((l) => l.name === cutLayer);
  if (!layer) throw new Error(`step ${step.kind}:${step.position} has no ${cutLayer} layer to cut at`);
  const cut = layer.first;
  if (!(cut > step.opFirst && cut < step.opEnd)) throw new Error(`cut ${cut} is outside step ${step.kind}:${step.position}`);
  const definedB = new Set();
  const carry = [];
  for (let i = cut; i < step.opEnd; i++) {
    const op = program.ops[i];
    for (const r of [op.a, op.b]) if (r >= 0 && !definedB.has(r) && !carry.includes(r)) carry.push(r);
    definedB.add(op.out);
  }
  const kv = new Set((step.stateIn ?? []).map((s) => s.reg));
  if (carry.some((r) => kv.has(r) || step.inputs.includes(r))) {
    throw new Error('stage B would read the KV cache or a requester input; this cut does not split the step');
  }
  if ((step.stateOut ?? []).some((s) => definedB.has(s.reg))) throw new Error('the step writes KV cache entries after the cut');
  if (!definedB.has(step.output)) throw new Error('the step output is not computed after the cut');
  return {
    cutLayer, cut,
    carry: carry.map((id) => ({ id, deg: program.regs[id].deg, level: program.regs[id].level })),
    layersA: step.layers.filter((l) => l.last < cut).map((l) => l.name),
    layersB: step.layers.filter((l) => l.first >= cut).map((l) => l.name),
  };
}
