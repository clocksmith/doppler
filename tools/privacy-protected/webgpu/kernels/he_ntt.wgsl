// he_ntt.wgsl
//
// Negacyclic NTT and inverse NTT, in place on `o`, batched over `limbs` consecutive limbs (limb i uses global
// modulus mod_base + i and its twiddle row). The transform is Lattigo's: Cooley-Tukey with powers of the primitive
// 2N-th root in bit-reversed order (forward, natural in / bit-reversed out) and Gentleman-Sande with the inverse
// powers (inverse), followed by N^-1. Every butterfly reduces fully, so outputs are canonical and equal Lattigo's.
//
// Stages whose butterfly distance t is at least TILE/2 run one dispatch per stage (ntt_fwd_global, ntt_inv_global);
// the remaining log2(TILE) stages run in one dispatch on a TILE-word tile in workgroup memory (ntt_fwd_local,
// ntt_inv_local). Requires N >= 2 * TILE / 2 = TILE, checked by the host.
//
// Uniforms: n, log_n, limbs, mod_base, o_off; global kernels also t = butterfly distance, log_t = log2(t);
// ntt_inv_global x0 = 1 multiplies the outputs by N^-1 (the last inverse stage); x1 = 1 uses modulus mod_base for
// every limb (several components of one limb, as in rescaling).

const TILE: u32 = 2048u;
const TILE_THREADS: u32 = 256u;

var<workgroup> tile: array<u32, 2048>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ntt_fwd_global(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_n = u.n >> 1u;
    let g = gid.x;
    if (g >= u.limbs * half_n) {
        return;
    }
    let limb = g >> (u.log_n - 1u);
    let bfly = g & (half_n - 1u);
    let i = bfly >> u.log_t;
    let off = bfly & (u.t - 1u);
    let j1 = u.o_off + limb * u.n + (i << (u.log_t + 1u)) + off;
    let j2 = j1 + u.t;
    let mi = select(u.mod_base + limb, u.mod_base, u.x1 == 1u);
    let md = mods[mi];
    let m = half_n >> u.log_t; // groups in this stage
    let w = tw[mi * u.n + m + i];
    let x = o[j1];
    let y = mont_mul(o[j2], w, md.x, md.y);
    o[j1] = add_mod(x, y, md.x);
    o[j2] = sub_mod(x, y, md.x);
}

@compute @workgroup_size(TILE_THREADS, 1, 1)
fn ntt_fwd_local(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tiles = u.n / TILE;
    let limb = wid.x / tiles;
    let blk = wid.x % tiles;
    let base = u.o_off + limb * u.n + blk * TILE;
    let mi = select(u.mod_base + limb, u.mod_base, u.x1 == 1u);
    let md = mods[mi];
    let row = mi * u.n;
    for (var k = 0u; k < TILE / TILE_THREADS; k = k + 1u) {
        tile[lid + k * TILE_THREADS] = o[base + lid + k * TILE_THREADS];
    }
    workgroupBarrier();
    var t = TILE >> 1u;
    var log_t = 10u;
    loop {
        let m = u.n >> (log_t + 1u);
        let groups_per_tile = (TILE >> 1u) >> log_t;
        for (var k = 0u; k < (TILE >> 1u) / TILE_THREADS; k = k + 1u) {
            let lb = lid + k * TILE_THREADS;
            let iloc = lb >> log_t;
            let off = lb & (t - 1u);
            let j1 = (iloc << (log_t + 1u)) + off;
            let j2 = j1 + t;
            let w = tw[row + m + blk * groups_per_tile + iloc];
            let x = tile[j1];
            let y = mont_mul(tile[j2], w, md.x, md.y);
            tile[j1] = add_mod(x, y, md.x);
            tile[j2] = sub_mod(x, y, md.x);
        }
        workgroupBarrier();
        if (t == 1u) {
            break;
        }
        t = t >> 1u;
        log_t = log_t - 1u;
    }
    for (var k = 0u; k < TILE / TILE_THREADS; k = k + 1u) {
        o[base + lid + k * TILE_THREADS] = tile[lid + k * TILE_THREADS];
    }
}

@compute @workgroup_size(TILE_THREADS, 1, 1)
fn ntt_inv_local(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tiles = u.n / TILE;
    let limb = wid.x / tiles;
    let blk = wid.x % tiles;
    let base = u.o_off + limb * u.n + blk * TILE;
    let mi = select(u.mod_base + limb, u.mod_base, u.x1 == 1u);
    let md = mods[mi];
    let row = mi * u.n;
    for (var k = 0u; k < TILE / TILE_THREADS; k = k + 1u) {
        tile[lid + k * TILE_THREADS] = o[base + lid + k * TILE_THREADS];
    }
    workgroupBarrier();
    var t = 1u;
    var log_t = 0u;
    loop {
        let h = u.n >> (log_t + 1u);
        let groups_per_tile = (TILE >> 1u) >> log_t;
        for (var k = 0u; k < (TILE >> 1u) / TILE_THREADS; k = k + 1u) {
            let lb = lid + k * TILE_THREADS;
            let iloc = lb >> log_t;
            let off = lb & (t - 1u);
            let j1 = (iloc << (log_t + 1u)) + off;
            let j2 = j1 + t;
            let w = tw[row + h + blk * groups_per_tile + iloc];
            let x = tile[j1];
            let y = tile[j2];
            tile[j1] = add_mod(x, y, md.x);
            tile[j2] = mont_mul(sub_mod(x, y, md.x), w, md.x, md.y);
        }
        workgroupBarrier();
        if (t == (TILE >> 1u)) {
            break;
        }
        t = t << 1u;
        log_t = log_t + 1u;
    }
    for (var k = 0u; k < TILE / TILE_THREADS; k = k + 1u) {
        o[base + lid + k * TILE_THREADS] = tile[lid + k * TILE_THREADS];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ntt_inv_global(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_n = u.n >> 1u;
    let g = gid.x;
    if (g >= u.limbs * half_n) {
        return;
    }
    let limb = g >> (u.log_n - 1u);
    let bfly = g & (half_n - 1u);
    let i = bfly >> u.log_t;
    let off = bfly & (u.t - 1u);
    let j1 = u.o_off + limb * u.n + (i << (u.log_t + 1u)) + off;
    let j2 = j1 + u.t;
    let mi = select(u.mod_base + limb, u.mod_base, u.x1 == 1u);
    let md = mods[mi];
    let h = half_n >> u.log_t;
    let w = tw[mi * u.n + h + i];
    let x = o[j1];
    let y = o[j2];
    var r1 = add_mod(x, y, md.x);
    var r2 = mont_mul(sub_mod(x, y, md.x), w, md.x, md.y);
    if (u.x0 == 1u) {
        r1 = mont_mul(r1, md.w, md.x, md.y);
        r2 = mont_mul(r2, md.w, md.x, md.y);
    }
    o[j1] = r1;
    o[j2] = r2;
}
