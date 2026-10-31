// Shared-slot down with the slot combine plus the layer's outer residual: one more
// binding, and the residual row added to the rounded combine as residual.wgsl adds
// it at unit scale. Routed/shared accumulation is the source's.
export const MOE_SHARED_SLOT_DOWN_PAIR_COMBINE_RESIDUAL_VARIANT = Object.freeze({
  id: 'moe-shared-slot-down-pair-combine-residual',
  source: 'src/gpu/kernels/moe_shared_slot_down_pair_combine_f32.wgsl',
  target: 'src/gpu/kernels/moe_shared_slot_down_pair_combine_residual_f32.wgsl',
  rules: [
    {
      type: 'literal',
      count: 1,
      from: '// Native GGUF shared-slot down projection fused with the slot combine: one workgroup\n'
        + '// per (row group, token) runs all TOP_K routed slots and the shared slot (slot TOP_K)\n'
        + '// for its rows. Each slot\'s row dot uses the pair geometry of\n',
      to: '// Native GGUF shared-slot down projection fused with the slot combine and the layer\'s\n'
        + '// outer residual: moe_shared_slot_down_pair_combine_f32.wgsl with one more binding.\n'
        + '// One workgroup per (row group, token) runs all TOP_K routed slots and the shared slot\n'
        + '// (slot TOP_K) for its rows. Each slot\'s row dot uses the pair geometry of\n',
    },
    {
      type: 'literal',
      count: 1,
      from: '// sigmoid-gated shared slot. Per-slot outputs never leave the workgroup.\n',
      to: '// sigmoid-gated shared slot. The rounded combine is then added to the residual row,\n'
        + '// as residual.wgsl adds the FFN output to it (unit scale). Per-slot outputs and the\n'
        + '// combine never leave the workgroup.\n',
    },
    {
      type: 'literal',
      count: 1,
      from: '@group(0) @binding(7) var<storage, read_write> output: array<f32>;\n',
      to: '@group(0) @binding(7) var<storage, read_write> output: array<f32>;\n'
        + '@group(0) @binding(8) var<storage, read> residual: array<f32>;\n',
    },
    {
      type: 'literal',
      count: 1,
      from: '        output[token * u.hidden_size + row] = routed + clamp_product(sigmoid(scalar_gate[token]) * shared_y);\n',
      to: '        let index = token * u.hidden_size + row;\n'
        + '        let combined = routed + clamp_product(sigmoid(scalar_gate[token]) * shared_y);\n'
        + '        output[index] = combined + residual[index];\n',
    },
  ],
});
