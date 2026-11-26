/**
 * The gated delta rule and gated RMSNorm for value heads [h0, h1) over n positions, shared by the requester's main
 * thread and its worker threads (no imports, so a Web Worker can load it too). Each value head's recurrence depends
 * only on its own state, so heads split across threads without changing any arithmetic.
 *
 *   S: [VH][dk][dv] recurrent state (float32, updated in place)
 *   q, k: [n][KH * dk] (l2-normalized; q already scaled), v, z, out: [n][VH * dv], g, beta: [n][VH]
 */
export function deltaRuleHeads(S, q, k, v, g, beta, z, gnorm, out, n, d, h0, h1) {
  const { dk, dv, VH, KH, eps } = d, rep = VH / KH, kd = KH * dk, vd = VH * dv;
  const kv = new Float64Array(dv), delta = new Float64Array(dv), acc = new Float64Array(dv);
  for (let t = 0; t < n; t++) {
    for (let hv = h0; hv < h1; hv++) {
      const hk = Math.floor(hv / rep), base = hv * dk * dv, eg = Math.exp(g[t * VH + hv]), bt = beta[t * VH + hv];
      const qo = t * kd + hk * dk, vo = t * vd + hv * dv;
      kv.fill(0); acc.fill(0);
      for (let i = 0; i < dk; i++) {
        const ki = k[qo + i], o = base + i * dv;
        for (let j = 0; j < dv; j++) { const s = (S[o + j] = S[o + j] * eg); kv[j] += s * ki; }
      }
      for (let j = 0; j < dv; j++) delta[j] = Math.fround((v[vo + j] - Math.fround(kv[j])) * bt);
      for (let i = 0; i < dk; i++) {
        const ki = k[qo + i], qi = q[qo + i], o = base + i * dv;
        for (let j = 0; j < dv; j++) { const s = (S[o + j] = S[o + j] + ki * delta[j]); acc[j] += s * qi; }
      }
      if (!gnorm) { for (let j = 0; j < dv; j++) out[vo + j] = acc[j]; continue; }
      // Gated RMSNorm of this head's output: norm(out) * w * silu(z).
      let s2 = 0;
      for (let j = 0; j < dv; j++) { const a = Math.fround(acc[j]); acc[j] = a; s2 += a * a; }
      const inv = Math.fround(1 / Math.sqrt(Math.fround(Math.fround(s2 / dv) + eps)));
      for (let j = 0; j < dv; j++) {
        const zz = z[vo + j];
        out[vo + j] = Math.fround(Math.fround(acc[j] * inv) * gnorm[j]) * Math.fround(zz / (1 + Math.exp(-zz)));
      }
    }
  }
}
