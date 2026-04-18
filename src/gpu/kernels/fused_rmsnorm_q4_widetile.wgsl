// Fused RMSNorm + Q4_K WideTile Matmul — F32 A, Q4_K B, F32 C
//
// Prologue fuses RMSNorm into the WideTile matmul:
//   normed[m, k] = (input[m, k] / rms(input[m, :])) * norm_weight[k]
//   output[m, n] = sum_k(normed[m, k] * dequant(B_q4k)[k, n])
//
// Targets the two Gemma-style pre-matmul norms:
//   input_norm → {q,k,v}_proj  (each q/k/v call runs norm internally)
//   pre_feedforward_norm → {gate,up}_proj
//
// Redundant norm work across q/k/v is negligible (norm: O(K) per row;
// matmul: O(K*N_per_WG) per row). Each saved standalone rmsnorm dispatch
// cuts ~0.85 ms of Dawn/Vulkan bubble — the target.
//
// Weight offset: when `RMS_NORM_OFFSET = true`, the stored norm weight
// encodes `(weight - 1.0)` (Gemma family convention). This kernel reads
// stored_w and applies `1.0 + stored_w` at use-site.
//
// Shared memory: tileA[TILE_M][QK_K] f32 = 4 KB per WG block (sliced
// across K in the existing WideTile loop). The full-row normed tensor
// is also cached in shared at `row_sq_sum` + `rms` derived per row;
// we compute sum-of-squares across K chunks in the same outer loop.
//
// Two-pass structure: pass 1 accumulates sum_sq across K, computes rms;
// pass 2 reuses the same tileA buffer to multiply normed*weight and
// dequant+matmul. Pass 2 reads K in the same block iteration order.
//
// Adapted from the iter-8 WideTile design.

enable f16;

const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;
const NUM_SUBBLOCK_PAIRS: u32 = 4u;
const U32S_PER_PAIR: u32 = 8u;

override TILE_M: u32 = 4u;
override TILE_N: u32 = 256u;
override RMS_NORM_OFFSET: bool = false;
override WEIGHT_IS_F16: bool = false;

const MAX_TILE_M: u32 = 8u;
const MAX_TILE_N: u32 = 256u;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    num_blocks_per_row: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

struct Q4KBlock {
    d_dmin: u32,
    scales: array<u32, 3>,
    qs: array<u32, 32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B_q4k: array<Q4KBlock>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> norm_weight: array<u32>;

var<workgroup> tileA: array<f32, MAX_TILE_M * QK_K>;
// Per-row reduction buffer for pass 1: [TILE_M][TILE_N] partial sum-of-squares.
// Each thread writes its per-row partial, then workgroup-parallel tree reduction
// collapses to shared_sum_sq[m][0]. Derived row_rms_recip is cached for pass 2.
var<workgroup> shared_sum_sq: array<array<f32, MAX_TILE_N>, MAX_TILE_M>;
var<workgroup> row_rms_recip: array<f32, MAX_TILE_M>;

fn unpack_f16_lo(packed: u32) -> f32 { return unpack2x16float(packed).x; }
fn unpack_f16_hi(packed: u32) -> f32 { return unpack2x16float(packed).y; }

// norm_weight binding is declared as array<u32> so it can carry either packed
// f32 values (bitcast) or packed f16 pairs (Gemma-family hidden weights).
// Mirrors the load_weight pattern in rmsnorm.wgsl.
fn load_norm_weight(idx: u32) -> f32 {
    if (WEIGHT_IS_F16) {
        let packed = norm_weight[idx >> 1u];
        let pair = unpack2x16float(packed);
        return select(pair.x, pair.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(norm_weight[idx]);
}

fn get_scale_byte(scales: array<u32, 3>, byte_idx: u32) -> u32 {
    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    return (scales[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn get_scale_min_k4(scales: array<u32, 3>, j: u32) -> vec2<u32> {
    var sc: u32;
    var mn: u32;
    if (j < 4u) {
        sc = get_scale_byte(scales, j) & 63u;
        mn = get_scale_byte(scales, j + 4u) & 63u;
    } else {
        let q_j = get_scale_byte(scales, j + 4u);
        let q_lo = get_scale_byte(scales, j - 4u);
        let q_hi = get_scale_byte(scales, j);
        sc = (q_j & 0xFu) | ((q_lo >> 6u) << 4u);
        mn = (q_j >> 4u) | ((q_hi >> 6u) << 4u);
    }
    return vec2<u32>(sc, mn);
}

fn load_normed_a_vec4(row_local: u32, base_elem: u32) -> vec4<f32> {
    // tileA at this point holds normed values (pass 2). The rms scaling
    // and norm_weight multiplication have already been applied when tileA
    // was populated for this block. See $main pass-2 section.
    let a_row_base = row_local * QK_K;
    return vec4<f32>(
        tileA[a_row_base + base_elem + 0u],
        tileA[a_row_base + base_elem + 1u],
        tileA[a_row_base + base_elem + 2u],
        tileA[a_row_base + base_elem + 3u]
    );
}

@compute @workgroup_size(TILE_N, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    if (TILE_M > MAX_TILE_M) { return; }

    let col_local = lid.x;
    let row_base = wg_id.y * TILE_M;
    let col_base = wg_id.x * TILE_N;
    let col = col_base + col_local;

    var results: array<f32, MAX_TILE_M>;
    for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
        results[i] = 0.0;
    }

    let num_blocks = u.num_blocks_per_row;

    // ===== PASS 1: workgroup-parallel sum-of-squares per row across all K =====
    // Each of TILE_N threads accumulates a per-thread partial across blocks,
    // then a tree reduction over TILE_N lanes finalises the sum per row. The
    // prior serialised 4-thread reduction was the iter-21 perf wall; this
    // restores full workgroup parallelism (log2(TILE_N)=8 reduction steps).
    var local_sum_sq: array<f32, MAX_TILE_M>;
    for (var m: u32 = 0u; m < TILE_M; m = m + 1u) {
        local_sum_sq[m] = 0.0;
    }

    // Streaming accumulation: thread col_local handles k_in_block == col_local
    // across every block. QK_K (256) == TILE_N (256), so the mapping is 1:1
    // and every lane picks up a K-value per block without serialisation.
    for (var b: u32 = 0u; b < num_blocks; b = b + 1u) {
        let global_k = b * QK_K + col_local;
        if (global_k < u.K) {
            for (var m: u32 = 0u; m < TILE_M; m = m + 1u) {
                let global_row = row_base + m;
                if (global_row < u.M) {
                    let v = A[global_row * u.K + global_k];
                    local_sum_sq[m] = local_sum_sq[m] + v * v;
                }
            }
        }
    }

    // Publish partials into shared memory for tree reduction.
    for (var m: u32 = 0u; m < TILE_M; m = m + 1u) {
        shared_sum_sq[m][col_local] = local_sum_sq[m];
    }
    workgroupBarrier();

    // Tree reduction over TILE_N lanes. All TILE_M rows are reduced in
    // parallel at each stride — 8 barriers total (log2(256)).
    for (var stride: u32 = TILE_N / 2u; stride > 0u; stride = stride >> 1u) {
        if (col_local < stride) {
            for (var m: u32 = 0u; m < TILE_M; m = m + 1u) {
                shared_sum_sq[m][col_local] =
                    shared_sum_sq[m][col_local]
                    + shared_sum_sq[m][col_local + stride];
            }
        }
        workgroupBarrier();
    }

    // Derive 1/rms per row. One thread per row reads the reduced total.
    if (col_local < TILE_M) {
        let mean_sq = shared_sum_sq[col_local][0] / f32(u.K);
        row_rms_recip[col_local] = 1.0 / sqrt(mean_sq + u.eps);
    }
    workgroupBarrier();

    // ===== PASS 2: reload A, apply rmsnorm + weight, matmul =====
    for (var b: u32 = 0u; b < num_blocks; b = b + 1u) {
        // Cooperative load of tileA raw (re-read from global — cheap vs
        // matmul work), apply norm in place, then matmul. Applying norm
        // in the same cooperative load avoids an extra barrier.
        let total_a_elems = TILE_M * QK_K;
        for (var idx: u32 = col_local; idx < total_a_elems; idx = idx + TILE_N) {
            let load_row_local = idx / QK_K;
            let load_k = idx % QK_K;
            let global_row = row_base + load_row_local;
            let global_k = b * QK_K + load_k;
            var raw: f32 = 0.0;
            if (global_row < u.M && global_k < u.K) {
                raw = A[global_row * u.K + global_k];
            }
            // rmsnorm: x * (1/rms)
            var normed = raw * row_rms_recip[load_row_local];
            // weight: multiply by norm_weight (with Gemma (1+w) offset).
            if (global_k < u.K) {
                let stored_w = load_norm_weight(global_k);
                let effective_w = select(stored_w, 1.0 + stored_w, RMS_NORM_OFFSET);
                normed = normed * effective_w;
            }
            tileA[idx] = normed;
        }
        workgroupBarrier();

        if (col < u.N) {
            let block = B_q4k[col * num_blocks + b];
            let d = unpack_f16_lo(block.d_dmin);
            let dmin = unpack_f16_hi(block.d_dmin);

            for (var p: u32 = 0u; p < NUM_SUBBLOCK_PAIRS; p = p + 1u) {
                let sm_even = get_scale_min_k4(block.scales, 2u * p);
                let sm_odd = get_scale_min_k4(block.scales, 2u * p + 1u);
                let scale_even = d * f32(sm_even.x);
                let min_even_scalar = dmin * f32(sm_even.y);
                let scale_odd = d * f32(sm_odd.x);
                let min_odd_scalar = dmin * f32(sm_odd.y);
                let min_even = vec4<f32>(min_even_scalar);
                let min_odd = vec4<f32>(min_odd_scalar);

                let pair_base_even = p * 64u;
                let pair_base_odd = p * 64u + SUBBLOCK_SIZE;
                let qs_base = p * U32S_PER_PAIR;

                for (var uidx: u32 = 0u; uidx < U32S_PER_PAIR; uidx = uidx + 1u) {
                    let packed = block.qs[qs_base + uidx];
                    let lower_u32 = unpack4xU8(packed & 0x0F0F0F0Fu);
                    let upper_u32 = unpack4xU8((packed >> 4u) & 0x0F0F0F0Fu);
                    let w_even = scale_even * vec4<f32>(lower_u32) - min_even;
                    let w_odd = scale_odd * vec4<f32>(upper_u32) - min_odd;

                    let elem_even = pair_base_even + uidx * 4u;
                    let elem_odd = pair_base_odd + uidx * 4u;

                    for (var m: u32 = 0u; m < TILE_M; m = m + 1u) {
                        let a_even = load_normed_a_vec4(m, elem_even);
                        let a_odd = load_normed_a_vec4(m, elem_odd);
                        results[m] = results[m] + dot(a_even, w_even) + dot(a_odd, w_odd);
                    }
                }
            }
        }
        workgroupBarrier();
    }

    if (col < u.N) {
        for (var m: u32 = 0u; m < TILE_M; m = m + 1u) {
            let row = row_base + m;
            if (row < u.M) {
                C[row * u.N + col] = results[m] * u.alpha;
            }
        }
    }
}
