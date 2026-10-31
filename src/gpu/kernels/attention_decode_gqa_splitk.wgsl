// Split-K (flash-decoding) single-query attention for f32 contiguous KV, head_dim 256.
// Outer shape follows Pooled engine/wgsl/attn_dec.js (license below): one workgroup per
// (split, kv head) serving every query head of
// that kv head, per-split partials merged by a combine pass. A split covers
// split_len = max(CHUNK, up64(ceil(kv_len / SPLITS))) positions, so the dispatch grid is
// fixed (SPLITS x num_kv_heads) and depends on no position; splits past kv_len exit.
//
// main: workgroup (split s, kv head h). K and V rows are read once for all GROUP query
// heads. Per 64-position chunk: scores (4 invocations per key, 64 dims each, reduced
// with subgroup shuffles in a fixed order), online softmax (32 invocations per query
// head), then V accumulation (invocation d owns output dim d for every query head).
// combine: workgroup = query head; merges the splits in split order.
// Every sum has a fixed order that depends only on the position: deterministic.
// MIT License
//
// Copyright (c) 2026 Nehanth Narendrula
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
enable subgroups;

const HD: u32 = 256u;
const CHUNK: u32 = 64u;
const NEG_INF: f32 = -3.402823e+38;

override SPLITS: u32 = 64u;
// Query heads per kv head; GROUP * 32 equals the workgroup size (validated on the host).
const GROUP: u32 = 8u;

struct Uniforms {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    seq_len: u32,
    scale: f32,
    is_causal: u32,
    start_pos: u32,
    attn_softcap: f32,
    sliding_window: u32,
    kv_len_source: u32,
    kv_start: u32,
    page_size: u32,
    kv_layout: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> kv_len_buffer: array<u32>;
@group(0) @binding(7) var<storage, read_write> partials: array<f32>;

var<workgroup> q_sh: array<vec4<f32>, 512>;      // GROUP (<= 8) x 64 vec4
var<workgroup> p_sh: array<f32, 512>;            // GROUP x CHUNK scores, then probabilities
var<workgroup> m_sh: array<f32, 8>;
var<workgroup> l_sh: array<f32, 8>;
var<workgroup> c_sh: array<f32, 8>;

fn kv_len() -> u32 {
    if (u.kv_len_source == 0u) { return u.kv_len; }
    return kv_len_buffer[0];
}

fn split_len(n: u32) -> u32 {
    let per = (n + SPLITS - 1u) / SPLITS;
    return max(CHUNK, ((per + CHUNK - 1u) / CHUNK) * CHUNK);
}

// partials: acc [num_heads][SPLITS][HD], then (m, l) [num_heads][SPLITS][2]
fn ml_base() -> u32 { return u.num_heads * SPLITS * HD; }

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let s = wg.x;
    let kvh = wg.y;
    let n_kv = kv_len();
    let len = split_len(n_kv);
    let k0 = s * len;
    if (k0 >= n_kv) { return; }
    let k1 = min(n_kv, k0 + len);
    let head0 = kvh * GROUP;

    for (var i = t; i < GROUP * 64u; i += 256u) {
        q_sh[i] = Q[head0 * 64u + i];
    }
    if (t < GROUP) { m_sh[t] = NEG_INF; l_sh[t] = 0.0; }
    workgroupBarrier();

    var acc: array<f32, 8>;
    for (var g = 0u; g < 8u; g++) { acc[g] = 0.0; }
    let key = t / 4u;
    let quarter = t % 4u;
    let row_vec4 = u.num_kv_heads * 64u;

    for (var c = k0; c < k1; c += CHUNK) {
        let n = min(CHUNK, k1 - c);
        // Scores: invocation (key, quarter) dots 64 dims of K with every query head.
        var dots: array<f32, 8>;
        for (var g = 0u; g < 8u; g++) { dots[g] = 0.0; }
        if (key < n) {
            let kb = (c + key) * row_vec4 + kvh * 64u + quarter * 16u;
            for (var i = 0u; i < 16u; i++) {
                let kv = K[kb + i];
                for (var g = 0u; g < GROUP; g++) {
                    dots[g] += dot(kv, q_sh[g * 64u + quarter * 16u + i]);
                }
            }
        }
        for (var g = 0u; g < GROUP; g++) {
            var d = dots[g];
            d += subgroupShuffleXor(d, 1u);
            d += subgroupShuffleXor(d, 2u);
            if (quarter == 0u) {
                p_sh[g * CHUNK + key] = select(NEG_INF, d * u.scale, key < n);
            }
        }
        workgroupBarrier();
        // Online softmax: 32 invocations per query head (GROUP * 32 == 256), two positions each.
        // Each invocation rewrites only the two scores it read.
        let g = t / 32u;
        let lane = t % 32u;
        let s0 = p_sh[g * CHUNK + lane];
        let s1 = p_sh[g * CHUNK + lane + 32u];
        var cmax = max(s0, s1);
        for (var o = 16u; o > 0u; o /= 2u) { cmax = max(cmax, subgroupShuffleXor(cmax, o)); }
        let m_old = m_sh[g];
        let m_new = max(m_old, cmax);
        let p0 = select(0.0, exp(s0 - m_new), lane < n);
        let p1 = select(0.0, exp(s1 - m_new), lane + 32u < n);
        var psum = p0 + p1;
        for (var o = 16u; o > 0u; o /= 2u) { psum += subgroupShuffleXor(psum, o); }
        let corr = exp(m_old - m_new);
        p_sh[g * CHUNK + lane] = p0;
        p_sh[g * CHUNK + lane + 32u] = p1;
        workgroupBarrier();
        if (lane == 0u) {
            m_sh[g] = m_new;
            l_sh[g] = l_sh[g] * corr + psum;
            c_sh[g] = corr;
        }
        workgroupBarrier();
        // V: invocation t owns output dim t of every query head.
        for (var gg = 0u; gg < GROUP; gg++) { acc[gg] *= c_sh[gg]; }
        let vb = kvh * HD + t;
        let vrow = u.num_kv_heads * HD;
        for (var j = 0u; j < n; j++) {
            let v = V[(c + j) * vrow + vb];
            for (var gg = 0u; gg < GROUP; gg++) { acc[gg] += p_sh[gg * CHUNK + j] * v; }
        }
        workgroupBarrier();
    }
    for (var g = 0u; g < GROUP; g++) {
        partials[((head0 + g) * SPLITS + s) * HD + t] = acc[g];
    }
    if (t < GROUP) {
        let mi = ml_base() + ((head0 + t) * SPLITS + s) * 2u;
        partials[mi] = m_sh[t];
        partials[mi + 1u] = l_sh[t];
    }
}

@compute @workgroup_size(256, 1, 1)
fn combine(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let h = wg.x;
    let n_kv = kv_len();
    let len = split_len(n_kv);
    let used = (n_kv + len - 1u) / len;
    var m = NEG_INF;
    for (var s = 0u; s < used; s++) { m = max(m, partials[ml_base() + (h * SPLITS + s) * 2u]); }
    var num = 0.0;
    var den = 0.0;
    for (var s = 0u; s < used; s++) {
        let mi = ml_base() + (h * SPLITS + s) * 2u;
        let w = exp(partials[mi] - m);
        den += w * partials[mi + 1u];
        num += w * partials[(h * SPLITS + s) * HD + t];
    }
    output[h * HD + t] = select(0.0, num / den, den > 0.0);
}
