// he_elementwise.wgsl
//
// Coefficient-wise RNS operations in the NTT domain. gid.x indexes limb * N + coefficient over `limbs` limbs
// (limb i uses global modulus mod_base + i); gid.y is the ciphertext component. Component k of an operand starts at
// off + k * stride (stride 0 broadcasts a single polynomial). Multipliers in `c` are in Montgomery form; additive
// constants in `c` are canonical.
//
// Uniforms: n, log_n, limbs, mod_base, a_*, b_*, o_*, c_off, c_stride; ew_pmul/ew_padd also x0 = log2(N / 128),
// the width of one constant block of a compact 64-slot plaintext.

struct Elem {
    q: u32,
    qinv: u32,
    r2: u32,
    limb: u32,
    a: u32,
    b: u32,
    o: u32,
    e: u32,
    valid: bool,
}

fn elem(gid: vec3<u32>) -> Elem {
    var r: Elem;
    r.valid = gid.x < u.limbs * u.n;
    let e = select(0u, gid.x, r.valid);
    r.limb = e >> u.log_n;
    let md = mods[u.mod_base + r.limb];
    r.q = md.x;
    r.qinv = md.y;
    r.r2 = md.z;
    r.e = e;
    r.a = u.a_off + gid.y * u.a_stride + e;
    r.b = u.b_off + gid.y * u.b_stride + e;
    r.o = u.o_off + gid.y * u.o_stride + e;
    return r;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    o[r.o] = a[r.a];
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    o[r.o] = add_mod(a[r.a], b[r.b], r.q);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_sub(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    o[r.o] = sub_mod(a[r.a], b[r.b], r.q);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_neg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    o[r.o] = sub_mod(0u, a[r.a], r.q);
}

// o = a * K[limb]
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_smul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    o[r.o] = mont_mul(a[r.a], c[u.c_off + r.limb], r.q, r.qinv);
}

// o = a + K[limb]
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_sadd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    o[r.o] = add_mod(a[r.a], c[u.c_off + r.limb], r.q);
}

// o = b + a * K[limb]
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_smac(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    o[r.o] = add_mod(b[r.b], mont_mul(a[r.a], c[u.c_off + r.limb], r.q, r.qinv), r.q);
}

fn compact_index(r: Elem) -> u32 {
    let j = r.e & (u.n - 1u);
    return u.c_off + r.limb * 128u + (j >> u.x0);
}

// o = a * pt (compact plaintext)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_pmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    o[r.o] = mont_mul(a[r.a], c[compact_index(r)], r.q, r.qinv);
}

// o = a + pt (compact plaintext)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_padd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    o[r.o] = add_mod(a[r.a], c[compact_index(r)], r.q);
}

// o += a * C (C: full polynomials in Montgomery form, component k at c_off + k * c_stride). Key inner products.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_mmac(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    let k = c[u.c_off + gid.y * u.c_stride + r.e];
    o[r.o] = add_mod(o[r.o], mont_mul(a[r.a], k, r.q, r.qinv), r.q);
}

// o = a * C (first digit of a key inner product).
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_mmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    let k = c[u.c_off + gid.y * u.c_stride + r.e];
    o[r.o] = mont_mul(a[r.a], k, r.q, r.qinv);
}

// In place: o = o * 2^32 mod q (canonical to Montgomery form; used once when keys are loaded).
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_to_mont(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    o[r.o] = mont_mul(o[r.o], r.r2, r.q, r.qinv);
}

// Degree-1 x degree-1 tensor product; gid.y selects the output component (0, 1, 2).
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_tensor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(vec3<u32>(gid.x, 0u, 0u));
    if (!r.valid) { return; }
    let a0 = a[u.a_off + r.e];
    let a1 = a[u.a_off + u.a_stride + r.e];
    let b0 = b[u.b_off + r.e];
    let b1 = b[u.b_off + u.b_stride + r.e];
    var v: u32;
    if (gid.y == 0u) {
        v = mont_mul(mont_mul(a0, b0, r.q, r.qinv), r.r2, r.q, r.qinv);
    } else if (gid.y == 1u) {
        let s = add_mod(mont_mul(a0, b1, r.q, r.qinv), mont_mul(a1, b0, r.q, r.qinv), r.q);
        v = mont_mul(s, r.r2, r.q, r.qinv);
    } else {
        v = mont_mul(mont_mul(a1, b1, r.q, r.qinv), r.r2, r.q, r.qinv);
    }
    o[u.o_off + gid.y * u.o_stride + r.e] = v;
}

// o = (a - b) * K[limb]: the last step of rescaling (K = q_l^-1) and of ModDown (K = P^-1).
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_sub_smul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    o[r.o] = mont_mul(sub_mod(a[r.a], b[r.b], r.q), c[u.c_off + r.limb], r.q, r.qinv);
}

// Automorphism in the NTT domain: o[j] = a[idx[j]], idx (N words) at c_off.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn ew_permute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = elem(gid);
    if (!r.valid) { return; }
    let j = r.e & (u.n - 1u);
    o[r.o] = a[u.a_off + gid.y * u.a_stride + (r.limb << u.log_n) + c[u.c_off + j]];
}
