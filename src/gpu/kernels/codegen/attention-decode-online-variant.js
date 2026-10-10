export const ATTENTION_DECODE_ONLINE_F16KV_VARIANT = {
    id: 'attention-decode-online-f16kv',
    source: 'src/gpu/kernels/attention_decode_online_f16.wgsl',
    target: 'src/gpu/kernels/attention_decode_online_f16kv.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: '// Online Decode Attention Kernel (f16 QKV + f16 output)',
        to: '// Online Decode Attention Kernel (f32 Q + f16 KV + f32 output)',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> Q: array<f16>;',
        to: '@group(0) @binding(1) var<storage, read> Q: array<f32>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(4) var<storage, read_write> output: array<f16>;',
        to: '@group(0) @binding(4) var<storage, read_write> output: array<f32>;',
      },
      {
        type: 'literal',
        count: 1,
        from: 'const NEG_INF = f16(-65504.0);\n',
        to: '',
      },
      {
        type: 'literal',
        count: 1,
        from: 'var<workgroup> shared_q: array<f16, MAX_HEAD_DIM>;',
        to: 'var<workgroup> shared_q: array<f32, MAX_HEAD_DIM>;',
      },
      {
        type: 'literal',
        count: 1,
        from: 'var<workgroup> shared_scores: array<f16, MAX_WORKGROUP_SIZE>;',
        to: 'var<workgroup> shared_scores: array<f32, MAX_WORKGROUP_SIZE>;',
      },
      {
        type: 'literal',
        count: 1,
        from: 'var<workgroup> sg_max: array<f16, MAX_SUBGROUPS>;',
        to: 'var<workgroup> sg_max: array<f32, MAX_SUBGROUPS>;',
      },
      {
        type: 'literal',
        count: 1,
        from: 'var<workgroup> sg_sum: array<f16, MAX_SUBGROUPS>;',
        to: 'var<workgroup> sg_sum: array<f32, MAX_SUBGROUPS>;',
      },
      {
        type: 'literal',
        count: 1,
        from: 'var<workgroup> global_max: f16;',
        to: 'var<workgroup> global_max: f32;',
      },
      {
        type: 'literal',
        count: 1,
        from: 'var<workgroup> global_sum: f16;',
        to: 'var<workgroup> global_sum: f32;',
      },
      {
        type: 'literal',
        count: 1,
        from: '            output[q_offset + out_dim0] = f16(0.0);',
        to: '            output[q_offset + out_dim0] = 0.0;',
      },
      {
        type: 'literal',
        count: 1,
        from: '            output[q_offset + out_dim1] = f16(0.0);',
        to: '            output[q_offset + out_dim1] = 0.0;',
      },
      {
        type: 'literal',
        count: 1,
        from: `    let scale = f16(u.scale);
    let softcap = f16(u.attn_softcap);
`,
        to: '',
      },
      {
        type: 'literal',
        count: 1,
        from: '    var running_max: f16 = NEG_INF;',
        to: '    var running_max: f32 = -3.402823e+38;',
      },
      {
        type: 'literal',
        count: 1,
        from: '    var running_sum: f16 = f16(0.0);',
        to: '    var running_sum: f32 = 0.0;',
      },
      {
        type: 'literal',
        count: 1,
        from: '    var out_accum0: f16 = f16(0.0);',
        to: '    var out_accum0: f32 = 0.0;',
      },
      {
        type: 'literal',
        count: 1,
        from: '    var out_accum1: f16 = f16(0.0);',
        to: '    var out_accum1: f32 = 0.0;',
      },
      {
        type: 'literal',
        count: 1,
        from: '        var score: f16 = NEG_INF;',
        to: '        var score: f32 = -3.402823e+38;',
      },
      {
        type: 'literal',
        count: 1,
        from: '                var dot: f16 = f16(0.0);',
        to: '                var dot: f32 = 0.0;',
      },
      {
        type: 'literal',
        count: 1,
        from: '                    let k0 = K[k_offset + d];',
        to: '                    let k0 = f32(K[k_offset + d]);',
      },
      {
        type: 'literal',
        count: 1,
        from: '                        let k1 = K[k_offset + d + 1u];',
        to: '                        let k1 = f32(K[k_offset + d + 1u]);',
      },
      {
        type: 'literal',
        count: 1,
        from: `                score = dot * scale;
                if (softcap > f16(0.0)) {
                    score = tanh(score / softcap) * softcap;
                }`,
        to: `                score = dot * u.scale;
                if (u.attn_softcap > 0.0) {
                    score = tanh(score / u.attn_softcap) * u.attn_softcap;
                }`,
      },
      {
        type: 'literal',
        count: 1,
        from: '        let chunk_max = subgroupMax(score);',
        to: '        var chunk_max = subgroupMax(score);',
      },
      {
        type: 'literal',
        count: 1,
        from: '            var m: f16 = NEG_INF;',
        to: '            var m: f32 = -3.402823e+38;',
      },
      {
        type: 'literal',
        count: 1,
        from: '        var exp_score: f16 = f16(0.0);',
        to: '        var exp_score: f32 = 0.0;',
      },
      {
        type: 'literal',
        count: 1,
        from: '        let chunk_sum = subgroupAdd(exp_score);',
        to: '        var chunk_sum = subgroupAdd(exp_score);',
      },
      {
        type: 'literal',
        count: 1,
        from: '            var s: f16 = f16(0.0);',
        to: '            var s: f32 = 0.0;',
      },
      {
        type: 'literal',
        count: 1,
        from: '                    out_accum0 = out_accum0 + shared_scores[score_idx] * V[v_base + out_dim0];',
        to: '                    out_accum0 = out_accum0 + shared_scores[score_idx] * f32(V[v_base + out_dim0]);',
      },
      {
        type: 'literal',
        count: 1,
        from: '                    out_accum1 = out_accum1 + shared_scores[score_idx] * V[v_base + out_dim1];',
        to: '                    out_accum1 = out_accum1 + shared_scores[score_idx] * f32(V[v_base + out_dim1]);',
      },
      {
        type: 'literal',
        count: 1,
        from: '    let inv_sum = select(f16(0.0), f16(1.0) / running_sum, running_sum > f16(0.0));',
        to: '    let inv_sum = select(0.0, 1.0 / running_sum, running_sum > 0.0);',
      },
      {
        type: 'literal',
        count: 1,
        from: '                var dot: f32 = 0.0;',
        to: '                var dot: f32 = 0.0;\n                var correction: f32 = 0.0;',
      },
      ...['0', '1'].map((suffix) => ({
        type: 'literal',
        count: 1,
        from: `dot = dot + q${suffix} * k${suffix};`,
        // Explicit fused operations retain the rounding needed by compensation.
        to: `let product${suffix} = fma(q${suffix}, k${suffix}, 0.0);
                    let total${suffix} = fma(1.0, dot, product${suffix});
                    let error${suffix} = select(fma(1.0, dot, fma(-1.0, total${suffix}, product${suffix})),
                        fma(1.0, product${suffix}, fma(-1.0, total${suffix}, dot)), abs(dot) >= abs(product${suffix}));
                    correction = fma(1.0, correction, error${suffix});
                    dot = total${suffix};`,
      })),
      {
        type: 'literal',
        count: 1,
        from: 'score = dot * u.scale;',
        to: 'score = fma(1.0, dot, correction) * u.scale;',
      },
    ],
  };
