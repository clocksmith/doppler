// AUTO-GENERATED from src/gpu/kernels/attention_prefill_query_tile_f32.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// F32 query/output and F16 KV prefill query tile. Existing attention uniform/cache/mask contract.
// Declared query tiles share one 16-key tile; each query uses a logical 32-lane
// subgroup partition, with eight Q/output components per lane. XOR offsets
// never cross partitions even when the physical subgroup is 64 lanes wide.
// No split-K partials: online (maximum, denominator, accumulator) stays local.

enable f16;
enable subgroups;

override QUERY_TILE: u32;
const KEY_TILE: u32 = 16u;
const PARTITION_LANES: u32 = 32u;
// Registry admission binds HEAD_DIM=256 and WORKGROUP_SIZE=QUERY_TILE*32.
// The key tile and per-lane register capacity remain fixed.
override HEAD_DIM: u32;
override WORKGROUP_SIZE: u32;
const LANE_REGISTER_CAPACITY: u32 = 8u;
const U32_MAX: u32 = 0xffffffffu;
var<workgroup> kv_tile: array<f32, KEY_TILE * HEAD_DIM>;

struct Uniforms {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    query_len: u32,
    scale: f32,
    is_causal: u32,
    start_pos: u32,  // Absolute position offset for causal masking
    attn_softcap: f32,    // Gemma 2: 50.0, 0 = disabled
    sliding_window: u32,  // Sliding window size (0 = disabled, >0 = window size)
    kv_len_source: u32,   // 0 = use uniform seq_len, 1 = use buffer
    kv_start: u32,
    page_size: u32,
    kv_layout: u32,
    bidirectional_span_start: u32,
    bidirectional_span_length: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> kv_len_buffer: array<u32>;
@group(0) @binding(6) var<storage, read> page_table: array<u32>;

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

fn is_bidirectional_span_visible(abs_query: u32, abs_key: u32) -> bool {
    if (u.bidirectional_span_length == 0u) {
        return false;
    }
    let span_start = u.bidirectional_span_start;
    let span_end = span_start + u.bidirectional_span_length;
    return abs_query >= span_start
        && abs_query < span_end
        && abs_key >= span_start
        && abs_key < span_end;
}

fn is_masked(query_pos: u32, key_pos: u32) -> bool {
    let abs_query = query_pos + u.start_pos;
    let abs_key = u.kv_start + key_pos;
    // Causal mask
    if (u.is_causal != 0u && abs_key > abs_query) {
        if (is_bidirectional_span_visible(abs_query, abs_key)) { return false; }
        return true;
    }
    // Sliding window mask
    if (u.sliding_window > 0u && abs_query >= u.sliding_window) {
        if (abs_key < abs_query - u.sliding_window + 1u) { return true; }
    }
    return false;
}

fn get_kv_pos(key_pos: u32) -> u32 {
    let abs_key = u.kv_start + key_pos;
    if (u.kv_layout == 1u && u.sliding_window > 0u) {
        return abs_key % u.sliding_window;
    }
    if (u.kv_layout == 2u) {
        let page_idx = abs_key / u.page_size;
        let in_page = abs_key - (page_idx * u.page_size);
        let phys_page = page_table[page_idx];
        return phys_page * u.page_size + in_page;
    }
    return abs_key;
}

fn get_kv_len() -> u32 {
    if (u.kv_len_source == 0u) {
        return u.seq_len;
    }
    return kv_len_buffer[0];
}

// KV positions this workgroup's outer loop visits. A causal dispatch without a
// bidirectional span masks every key after the workgroup's last valid query, so
// the loop stops at that query's absolute position (inclusive); a query before
// kv_start sees no key. A span can expose later keys, so it keeps the full loop,
// as does a position that would overflow u32. Per-key masks still apply inside.
fn causal_kv_end(query_block: u32, kv_len: u32) -> u32 {
    if (u.is_causal == 0u || u.bidirectional_span_length != 0u) { return kv_len; }
    let first = query_block * QUERY_TILE;
    if (first >= u.query_len) { return 0u; }
    let last = first + min(QUERY_TILE - 1u, u.query_len - 1u - first);
    if (u.start_pos > U32_MAX - last) { return kv_len; }
    let abs_last = last + u.start_pos;
    if (abs_last < u.kv_start) { return 0u; }
    let through = abs_last - u.kv_start;
    if (through >= kv_len) { return kv_len; }
    return through + 1u;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_lane: u32,
) {
    let head = wg_id.x % u.num_heads;
    let query_block = wg_id.x / u.num_heads;
    let logical_index = subgroup_id * subgroup_size + subgroup_lane;
    let lane = subgroup_lane % PARTITION_LANES;
    let query = query_block * QUERY_TILE + logical_index / PARTITION_LANES;
    let valid = query < u.query_len;
    let kv_head = get_kv_head_idx(head);
    let kv_len = get_kv_len();
    let kv_end = causal_kv_end(query_block, kv_len);
    var q: array<f32, LANE_REGISTER_CAPACITY>;
    var acc: array<f32, LANE_REGISTER_CAPACITY>;
    if (valid) {
        let base = (query * u.num_heads + head) * HEAD_DIM;
        for (var d = 0u; d < LANE_REGISTER_CAPACITY; d++) {
            q[d] = Q[base + lane + d * PARTITION_LANES];
        }
    }
    var maximum = -3.402823e+38f;
    var denominator = 0.0f;
    // Uniform loop bounds keep both shared-memory barriers and shuffles valid
    // for tail queries and completely masked rows.
    for (var start = 0u; start < kv_end; start += KEY_TILE) {
        for (var i = local_index; i < KEY_TILE * HEAD_DIM; i += WORKGROUP_SIZE) {
            let key = start + i / HEAD_DIM;
            var value = 0.0f;
            if (key < kv_len) {
                let base = (get_kv_pos(key) * u.num_kv_heads + kv_head) * HEAD_DIM;
                value = f32(K[base + i % HEAD_DIM]);
            }
            kv_tile[i] = value;
        }
        workgroupBarrier();
        var probabilities: array<f32, KEY_TILE>;
        var block_max = maximum;
        for (var k = 0u; k < KEY_TILE; k++) {
            var score = 0.0f;
            for (var d = 0u; d < LANE_REGISTER_CAPACITY; d++) {
                score += q[d] * kv_tile[k * HEAD_DIM + lane + d * PARTITION_LANES];
            }
            // All invocations execute all shuffles, including inactive rows.
            score += subgroupShuffleXor(score, 16u);
            score += subgroupShuffleXor(score, 8u);
            score += subgroupShuffleXor(score, 4u);
            score += subgroupShuffleXor(score, 2u);
            score += subgroupShuffleXor(score, 1u);
            score *= u.scale;
            if (u.attn_softcap > 0.0) {
                score = tanh(score / u.attn_softcap) * u.attn_softcap;
            }
            probabilities[k] = score;
            if (valid && start + k < kv_len && !is_masked(query, start + k)) {
                block_max = max(block_max, score);
            }
        }
        let correction = exp(maximum - block_max);
        denominator *= correction;
        for (var k = 0u; k < KEY_TILE; k++) {
            var probability = 0.0f;
            if (valid && start + k < kv_len && !is_masked(query, start + k)) {
                probability = exp(probabilities[k] - block_max);
            }
            probabilities[k] = probability;
            denominator += probability;
        }
        for (var d = 0u; d < LANE_REGISTER_CAPACITY; d++) { acc[d] *= correction; }
        maximum = block_max;
        // Every reader finishes K before the same tile is overwritten by V.
        workgroupBarrier();
        for (var i = local_index; i < KEY_TILE * HEAD_DIM; i += WORKGROUP_SIZE) {
            let key = start + i / HEAD_DIM;
            var value = 0.0f;
            if (key < kv_len) {
                let base = (get_kv_pos(key) * u.num_kv_heads + kv_head) * HEAD_DIM;
                value = f32(V[base + i % HEAD_DIM]);
            }
            kv_tile[i] = value;
        }
        workgroupBarrier();
        for (var k = 0u; k < KEY_TILE; k++) {
            for (var d = 0u; d < LANE_REGISTER_CAPACITY; d++) {
                acc[d] += probabilities[k] * kv_tile[k * HEAD_DIM + lane + d * PARTITION_LANES];
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        let base = (query * u.num_heads + head) * HEAD_DIM;
        for (var d = 0u; d < LANE_REGISTER_CAPACITY; d++) {
            var value = 0.0f;
            if (denominator > 0.0) { value = acc[d] / denominator; }
            output[base + lane + d * PARTITION_LANES] = value;
        }
    }
}
