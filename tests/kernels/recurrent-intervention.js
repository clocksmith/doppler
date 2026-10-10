import assert from 'node:assert/strict';

export function recurrentInterventionOperands(reference, intervention) {
  const groups = { normalization: ['qScale', 'kScale', 'invRms'],
    gates: ['beta', 'decay', 'gate'], decay: ['decay'], state: ['updatedState'] };
  const selections = { normalization: ['normalization'], gates: ['gates'], state: ['state'],
    decay: ['decay'], 'normalization-gates': ['normalization', 'gates'], all: ['normalization', 'gates', 'state'] };
  assert(selections[intervention], 'Unknown recurrent intervention');
  const names = selections[intervention].flatMap(name => groups[name]);
  const fields = {}; let length = 0;
  for (const name of names) {
    const size = reference.layout.fields[name].length;
    fields[name] = { offset: length, length: size }; length += size;
  }
  const values = new Float32Array(length);
  for (const name of names) {
    const f = reference.layout.fields[name];
    values.set(reference.trace.subarray(f.offset, f.offset + f.length), fields[name].offset);
  }
  return { values, layout: { fields, length } };
}

/** Causal experiment only: inject Float64-reference results rounded to F32.
 * This is deliberately not a production implementation or a CPU fallback. */
export function interveneRecurrentShader(source, layout, intervention) {
  assert(['normalization', 'gates', 'decay', 'state', 'normalization-gates', 'all'].includes(intervention));
  const at = (name, index) => `reference_values[${layout.fields[name].offset}u + ${index}]`;
  const replace = (original, replacement) => {
    assert.equal(source.split(original).length, 2, `Ambiguous intervention boundary: ${original}`);
    source = source.replace(original, replacement);
  };
  if (['normalization', 'normalization-gates', 'all'].includes(intervention)) {
    for (const [original, name, variable] of [
      ['head_scale * inverse_root_refined(shared_sq[0] + params.qk_l2norm_eps)', 'qScale', 'q_norm_scale'],
      ['inverse_root_refined(shared_sq[0] + params.qk_l2norm_eps)', 'kScale', 'k_norm_scale'],
      ['inverse_root_refined(shared_sq[0] / f32(head_v_dim) + params.rms_norm_eps)', 'invRms', 'inv_rms'],
    ]) replace(`let ${variable} = ${original};`, `let ${variable} = ${at(name, 'ab_row_base')};`);
  }
  if (['gates', 'normalization-gates', 'all'].includes(intervention)) {
    replace('let beta = sigmoid_refined(f32(b_proj[b_index]));', `let beta = ${at('beta', 'ab_row_base')};`);
    replace('let gate = silu(f32(z_proj[z_index]));', `let gate = ${at('gate', 'out_row_base + vd')};`);
  }
  if (['gates', 'decay', 'normalization-gates', 'all'].includes(intervention)) {
    replace('let g_exp = exp_refined(g);', `let g_exp = ${at('decay', 'ab_row_base')};`);
  }
  if (['state', 'all'].includes(intervention)) {
    const index = 'ab_row_base * head_k_dim * head_v_dim + kd * head_v_dim + vd';
    replace('recurrent_state[state_idx] = fma(k_normed, delta, recurrent_state[state_idx]);',
      `recurrent_state[state_idx] = ${at('updatedState', index)};`);
  }
  return source + '\n@group(0) @binding(10) var<storage, read> reference_values: array<f32>;\n';
}
