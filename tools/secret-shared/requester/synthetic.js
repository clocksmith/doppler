/**
 * Requester-only measurement at a model's shape without its weights: a synthetic Qwen 3.5 source (random small
 * norm, convolution and gate weights; the embedding read row by row from a file of the real size, which may be
 * sparse), and a matching plan for null operators (null-operator.js) that answer each round with shares of the right
 * size. The requester then does all of its real per-token work: encoding, PRG, sharing, recombination and every
 * nonlinear step. It is not a model run and its outputs mean nothing; it measures requester CPU, memory and bytes.
 */
import fs from 'node:fs';

export const SHAPES = {
  // From the RDRR manifests (0.8B) and ouroboros route3 cost-model.py (27B: qwen-3-6-27b-q4k-ehaf16).
  '0.8b': { layers: 24, hidden: 1024, inter: 3584, heads: 8, kvHeads: 2, headDim: 256, vocab: 248320, linKH: 16, linVH: 16, linDK: 128, linDV: 128, convK: 4 },
  '27b': { layers: 64, hidden: 5120, inter: 17408, heads: 24, kvHeads: 4, headDim: 256, vocab: 248320, linKH: 16, linVH: 48, linDK: 128, linDV: 128, convK: 4 }
};

export function syntheticConfig(name) {
  const s = SHAPES[name];
  return { ...s, layerTypes: Array.from({ length: s.layers }, (_, l) => ((l + 1) % 4 === 0 ? 'full_attention' : 'linear_attention')),
    eps: 1e-6, ropeTheta: 10000000, rotaryDim: s.headDim / 4, eos: [-1] };
}

/** The plan's groups (ids, widths, output rows), split into `stages` contiguous stages, head on the last. */
export function syntheticPlan(c, stages = 2, fx = 20, fw = 24) {
  const t = (name, rows, cols) => ({ name, shape: [rows, cols] });
  const kd = c.linKH * c.linDK, vd = c.linVH * c.linDV;
  const per = Math.ceil(c.layers / stages);
  return { schema: 'doppler.secret-shared-plan/v1', modelId: 'synthetic', fx, fw, stages: Array.from({ length: stages }, (_, s) => {
    const groups = [];
    for (let l = s * per; l < Math.min(c.layers, (s + 1) * per); l++) {
      if (c.layerTypes[l] === 'linear_attention') {
        groups.push({ id: `L${l}.in`, in: c.hidden, tensors: [t('qkv', 2 * kd + vd, c.hidden), t('z', vd, c.hidden), t('b', c.linVH, c.hidden), t('a', c.linVH, c.hidden)] });
        groups.push({ id: `L${l}.out`, in: vd, tensors: [t('out', c.hidden, vd)] });
      } else {
        groups.push({ id: `L${l}.in`, in: c.hidden, tensors: [t('q', 2 * c.heads * c.headDim, c.hidden), t('k', c.kvHeads * c.headDim, c.hidden), t('v', c.kvHeads * c.headDim, c.hidden)] });
        groups.push({ id: `L${l}.out`, in: c.heads * c.headDim, tensors: [t('o', c.hidden, c.heads * c.headDim)] });
      }
      groups.push({ id: `L${l}.up`, in: c.hidden, tensors: [t('gate', c.inter, c.hidden), t('up', c.inter, c.hidden)] });
      groups.push({ id: `L${l}.down`, in: c.inter, tensors: [t('down', c.hidden, c.inter)] });
    }
    if (s === stages - 1) groups.push({ id: 'head', in: c.hidden, tensors: [t('head', c.vocab, c.hidden)] });
    return { index: s, layers: [s * per, Math.min(c.layers, (s + 1) * per) - 1], groups };
  }) };
}

export function syntheticSource(c, embeddingFile) {
  let seed = 12345;
  const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648) * 2 - 1;
  const shapes = {
    input_layernorm: [c.hidden], post_attention_layernorm: [c.hidden], 'linear_attn.conv1d': [2 * c.linKH * c.linDK + c.linVH * c.linDV, 1, c.convK],
    'linear_attn.A_log': [c.linVH], 'linear_attn.dt_bias': [c.linVH], 'linear_attn.norm': [c.linDV], 'self_attn.q_norm': [c.headDim], 'self_attn.k_norm': [c.headDim]
  };
  let held = 0, bytes = 0;
  const fd = embeddingFile ? fs.openSync(embeddingFile, 'r') : null;
  const row = new Uint8Array(2 * c.hidden), view = new DataView(row.buffer);
  return {
    tensor(name) {
      const key = Object.keys(shapes).find(k => name.endsWith(`.${k}.weight`) || name.endsWith(`.${k}`)) ?? (name.endsWith('norm.weight') ? 'input_layernorm' : null);
      if (!key) throw Error(`synthetic: ${name}`);
      const shape = shapes[key], n = shape.reduce((a, b) => a * b, 1), values = new Float32Array(n);
      for (let i = 0; i < n; i++) values[i] = key.includes('A_log') ? Math.log(1 + 8 * Math.abs(rand())) : 0.1 * rand();
      held++; bytes += 4 * n;
      return { dtype: 'F32', shape, bytes: new Uint8Array(values.buffer) };
    },
    embeddingRow(id, dst) {
      if (fd === null) { dst.fill(0); return; }
      fs.readSync(fd, row, 0, row.length, id * row.length);
      for (let i = 0; i < c.hidden; i++) dst[i] = view.getUint16(2 * i, true) / 65536;
    },
    heldTensors: () => held, heldBytes: () => bytes
  };
}
