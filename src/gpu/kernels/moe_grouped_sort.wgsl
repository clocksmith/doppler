// Adapted from Pooled engine/wgsl/moe_group.js, revision
// 6b4a24f73cce7ef9a9e3ab35b9a0d4c09b51dfff: stable counting sort and F32 row tiles.
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
// 
override WORKGROUP_SIZE: u32;
override GROUP_SIZE: u32;
override MAX_EXPERTS: u32;
struct Uniforms {
    num_routes: u32, num_experts: u32, top_k: u32, pair_offset: u32,
    gu_groups: u32, down_groups: u32, _pad0: u32, _pad1: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> groups: array<u32>;
@group(0) @binding(3) var<storage, read_write> indirect: array<u32>;
var<workgroup> histogram: array<atomic<u32>, MAX_EXPERTS>;
var<workgroup> masks: array<atomic<u32>, MAX_EXPERTS>;
var<workgroup> bases: array<u32, MAX_EXPERTS>;
var<workgroup> placed: array<u32, MAX_EXPERTS>;
var<workgroup> chunks: array<u32, MAX_EXPERTS>;
var<workgroup> pairs: array<u32, MAX_EXPERTS>;
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(local_invocation_index) t: u32) {
    atomicStore(&histogram[t], 0u);
    placed[t] = 0u;
    workgroupBarrier();
    for (var i = t; i < u.num_routes; i += WORKGROUP_SIZE) {
        atomicAdd(&histogram[indices[i]], 1u);
    }
    workgroupBarrier();
    let count = atomicLoad(&histogram[t]);
    let count_chunks = (count + GROUP_SIZE - 1u) / GROUP_SIZE;
    chunks[t] = count_chunks;
    pairs[t] = count;
    workgroupBarrier();
    for (var offset = 1u; offset < MAX_EXPERTS; offset <<= 1u) {
        var c = chunks[t]; var p = pairs[t];
        if (t >= offset) { c += chunks[t - offset]; p += pairs[t - offset]; }
        workgroupBarrier();
        chunks[t] = c; pairs[t] = p;
        workgroupBarrier();
    }
    let chunk_base = chunks[t] - count_chunks;
    bases[t] = pairs[t] - count;
    workgroupBarrier();
    // Actual top-k output has distinct experts within a token. A round's bitmask
    // therefore gives a stable pair position without an order-dependent atomic append.
    let columns_per_round = min(28u, WORKGROUP_SIZE / u.top_k);
    let local_column = t / u.top_k;
    let slot = t % u.top_k;
    for (var c0 = 0u; c0 < u.num_routes / u.top_k; c0 += columns_per_round) {
        let col = c0 + local_column;
        let valid = local_column < columns_per_round && col < u.num_routes / u.top_k;
        let route = col * u.top_k + slot;
        var expert = 0u;
        if (valid) { expert = indices[route]; atomicStore(&masks[expert], 0u); }
        workgroupBarrier();
        if (valid) { atomicOr(&masks[expert], 1u << local_column); }
        workgroupBarrier();
        var mask = 0u;
        if (valid) {
            mask = atomicLoad(&masks[expert]);
            groups[u.pair_offset + bases[expert] + placed[expert]
                + countOneBits(mask & ((1u << local_column) - 1u))] = route;
        }
        workgroupBarrier();
        if (valid && (mask >> (local_column + 1u)) == 0u) { placed[expert] += countOneBits(mask); }
        workgroupBarrier();
    }
    for (var k = 0u; k * GROUP_SIZE < count; k++) {
        let at = (chunk_base + k) * 4u;
        groups[at] = bases[t] + k * GROUP_SIZE;
        groups[at + 1u] = min(GROUP_SIZE, count - k * GROUP_SIZE);
        groups[at + 2u] = t;
        groups[at + 3u] = 0u;
    }
    if (t == 0u) {
        indirect[0] = u.gu_groups; indirect[1] = chunks[MAX_EXPERTS - 1u]; indirect[2] = 1u;
        indirect[3] = u.down_groups; indirect[4] = chunks[MAX_EXPERTS - 1u]; indirect[5] = 1u;
    }
}
