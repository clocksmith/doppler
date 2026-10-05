import assert from 'node:assert/strict';

/** Test-only readback of the unchanged recurrent kernel. Always compare the
 * resulting output/state with an uninstrumented replay on the same device. */
export function observeRecurrentShader(source, layout, selectedStages = Object.keys(layout.fields)) {
  assert(selectedStages.length && selectedStages.every(name => layout.fields[name]), 'Unknown observation stage');
  const at = (name, index) => `recurrent_trace[${layout.fields[name].offset}u + ${index}]`;
  const scalar = name => at(name, 'ab_row_base');
  const vector = name => at(name, 'out_row_base + vd');
  const matrix = name => at(name, 'ab_row_base * head_k_dim * head_v_dim + kd * head_v_dim + vd');
  const after = (anchor, observation) => {
    const relevant = Object.entries(layout.fields).some(([name, field]) => selectedStages.includes(name)
      && observation.includes(`recurrent_trace[${field.offset}u +`));
    if (!relevant) return;
    assert.equal(source.split(anchor).length, 2, `Ambiguous recurrent observation anchor: ${anchor}`);
    source = source.replace(anchor, `${anchor}\n${observation}`);
  };
  after('let q_norm_scale = head_scale / sqrt(shared_sq[0] + params.qk_l2norm_eps);',
    `    if (vd == 0u) { ${scalar('qScale')} = q_norm_scale; }`);
  after('let k_norm_scale = inverseSqrt(shared_sq[0] + params.qk_l2norm_eps);',
    `    if (vd == 0u) { ${scalar('kScale')} = k_norm_scale; }`);
  after('kv_mem = kv_mem + recurrent_state[state_idx] * k_normed;',
    `        if (vd == 0u) { ${at('normalizedK', 'ab_row_base * head_k_dim + kd')} = k_normed; }`);
  after('out_value = out_value + recurrent_state[state_idx] * q_normed;',
    `        if (vd == 0u) { ${at('normalizedQ', 'ab_row_base * head_k_dim + kd')} = q_normed; }`);
  for (const [name, value] of [['beta', 'beta'], ['logDecay', 'g'], ['decay', 'g_exp']]) {
    after('let g_exp = exp(g);', `    if (vd == 0u) { ${scalar(name)} = ${value}; }`);
  }
  after('recurrent_state[state_idx] = recurrent_state[state_idx] * g_exp;',
    `        ${matrix('decayedState')} = recurrent_state[state_idx];`);
  after('let delta = (conv_out[conv_row_base + v_base + vd] - kv_mem) * beta;',
    `      ${vector('memory')} = kv_mem;`);
  after('let delta = (conv_out[conv_row_base + v_base + vd] - kv_mem) * beta;',
    `      ${vector('correction')} = delta;`);
  after('recurrent_state[state_idx] = recurrent_state[state_idx] + k_normed * delta;',
    `        ${matrix('updatedState')} = recurrent_state[state_idx];`);
  after('output[out_row_base + vd] = out_value;', `      ${vector('rawOutput')} = out_value;`);
  after('let inv_rms = inverseSqrt(shared_sq[0] / f32(head_v_dim) + params.rms_norm_eps);',
    `    if (vd == 0u) { ${scalar('invRms')} = inv_rms; }`);
  after('let gate = silu(f32(z_proj[z_index]));', `      ${vector('gate')} = gate;`);
  after('output[out_row_base + vd] = (output[out_row_base + vd] * inv_rms) * norm_weight[norm_index] * gate;',
    `      ${vector('gatedOutput')} = output[out_row_base + vd];`);
  return source + '\n@group(0) @binding(10) var<storage, read_write> recurrent_trace: array<f32>;\n';
}
