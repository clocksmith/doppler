// he_basis.wgsl
//
// RNS basis extension (coefficient domain), as in Lattigo's DecomposeAndSplit (Q digit -> Q and P) and ModUpPtoQ
// (P -> Q), plus the first step of rescaling.
//
// A conversion from source moduli D = {d_0..d_{k-1}} with product M lifts the centred value x in (-M/2, M/2]:
//   y_i = (x_i + floor(M/2)) * (M/d_i)^-1 mod d_i,  v = floor(sum_i y_i / d_i),
//   out_t = sum_i y_i * (M/d_i mod p_t) + v * (-M mod p_t) - floor(M/2) mod p_t.
// Lattigo estimates v in float64; here it is computed exactly enough in 96-bit fixed point
// (sum of y_i * floor(2^96 / d_i)), which differs from the exact floor only when the fraction is within k * 2^-64 of
// an integer. Tables (built by rns-tables.js) at c_off: (M/d_i)^-1 in Montgomery form [k], floor(M/2) mod d_i [k],
// floor(2^96/d_i) as three words [3k], then for every global modulus g at c_off + 5k + g*(k+2):
// M/d_i mod p_g in Montgomery form [k], -M mod p_g in Montgomery form, floor(M/2) mod p_g.

// conv_y: per coefficient j: y_i into o[o_off + i*N + j], v into o[o_off + k*N + j]. Sources at a_off + i*N,
// moduli mod_base + i; t = k.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn conv_y(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= u.n) {
        return;
    }
    let k = u.t;
    var acc0 = 0u;
    var acc1 = 0u;
    var acc2 = 0u;
    var acc3 = 0u;
    for (var i = 0u; i < k; i = i + 1u) {
        let md = mods[u.mod_base + i];
        let x = a[u.a_off + i * u.n + j];
        let y = mont_mul(add_mod(x, c[u.c_off + k + i], md.x), c[u.c_off + i], md.x, md.y);
        o[u.o_off + i * u.n + j] = y;
        let f = u.c_off + 2u * k + 3u * i;
        let p0 = mul_wide(y, c[f]);
        let p1 = mul_wide(y, c[f + 1u]);
        let p2 = mul_wide(y, c[f + 2u]);
        // y * F as four words (w0, w1, w2, w3)
        let w0 = p0.x;
        let w1 = p0.y + p1.x;
        let c1 = select(0u, 1u, w1 < p0.y);
        let w2a = p1.y + p2.x;
        let c2a = select(0u, 1u, w2a < p1.y);
        let w2 = w2a + c1;
        let c2b = select(0u, 1u, w2 < w2a);
        let w3 = p2.y + c2a + c2b;
        // acc += w
        let s0 = acc0 + w0;
        let k0 = select(0u, 1u, s0 < acc0);
        let s1a = acc1 + w1;
        let k1a = select(0u, 1u, s1a < acc1);
        let s1 = s1a + k0;
        let k1b = select(0u, 1u, s1 < s1a);
        let s2a = acc2 + w2;
        let k2a = select(0u, 1u, s2a < acc2);
        let s2 = s2a + k1a + k1b;
        let k2b = select(0u, 1u, s2 < s2a);
        acc0 = s0;
        acc1 = s1;
        acc2 = s2;
        acc3 = acc3 + w3 + k2a + k2b;
    }
    o[u.o_off + k * u.n + j] = acc3;
}

// conv_sum: gid.x over limbs * N; target limb l uses global modulus mod_base + l and writes o[o_off + l*N + j].
// y and v come from a at a_off (conv_y's output); t = k.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn conv_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.limbs * u.n) {
        return;
    }
    let limb = gid.x >> u.log_n;
    let j = gid.x & (u.n - 1u);
    let g = u.mod_base + limb;
    let md = mods[g];
    let k = u.t;
    let rec = u.c_off + 5u * k + g * (k + 2u);
    var acc = 0u;
    for (var i = 0u; i < k; i = i + 1u) {
        acc = add_mod(acc, mont_mul(a[u.a_off + i * u.n + j], c[rec + i], md.x, md.y), md.x);
    }
    let v = a[u.a_off + k * u.n + j];
    acc = add_mod(acc, mont_mul(v, c[rec + k], md.x, md.y), md.x);
    o[u.o_off + limb * u.n + j] = sub_mod(acc, c[rec + k + 1u], md.x);
}

// conv_copy1: a one-prime digit (Lattigo's copy path): the centred residue mod d (global modulus log_t), reduced
// into every target modulus. Source at a_off + j.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn conv_copy1(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.limbs * u.n) {
        return;
    }
    let limb = gid.x >> u.log_n;
    let j = gid.x & (u.n - 1u);
    let q = mods[u.mod_base + limb].x;
    let d = mods[u.log_t].x;
    let x = a[u.a_off + j];
    var r: u32;
    if (x >= (d >> 1u)) {
        r = sub_mod(0u, reduce_below(d - x, q), q);
    } else {
        r = reduce_below(x, q);
    }
    o[u.o_off + limb * u.n + j] = r;
}

// rescale_prep: a holds the last limb (index log_t) of each component after its inverse NTT, at
// a_off + comp * a_stride. For every lower limb i: ((x + h) mod q_l - h) mod q_i with h = (q_l - 1) / 2.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rescale_prep(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.limbs * u.n) {
        return;
    }
    let limb = gid.x >> u.log_n;
    let j = gid.x & (u.n - 1u);
    let q = mods[u.mod_base + limb].x;
    let ql = mods[u.log_t].x;
    let h = (ql - 1u) >> 1u;
    let t = add_mod(a[u.a_off + gid.y * u.a_stride + j], h, ql);
    o[u.o_off + gid.y * u.o_stride + gid.x] = sub_mod(reduce_below(t, q), reduce_below(h, q), q);
}
