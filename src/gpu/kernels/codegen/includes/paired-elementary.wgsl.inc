// Paired elementary arithmetic. Requires compensated-f32.wgsl.inc.
// Finite normalized input pairs; finite intermediate sums/products. Division
// requires nonzero denominator, sqrt nonnegative input, and log positive input.
// Exp range reduction is bounded to finite binary32 exponent range. These are
// arithmetic domains, not model-selection policy. No extended exponent format.
fn pair_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return pair_add(pair_mul_scalar(a, b.x), pair_mul_scalar(a, b.y));
}
fn pair_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let first = a.x / b.x;
    var result = vec2<f32>(first, 0.0);
    for (var i = 0u; i < 2u; i++) {
        let residual = pair_add(a, -pair_mul(result, b));
        result = pair_add(result, vec2<f32>(residual.x / b.x, 0.0));
    }
    return result;
}
fn pair_sqrt(a: vec2<f32>) -> vec2<f32> {
    if (all(a == vec2<f32>(0.0))) { return a; }
    var result = vec2<f32>(sqrt(a.x), 0.0);
    for (var i = 0u; i < 2u; i++) {
        let residual = pair_add(a, -pair_mul(result, result));
        result = pair_add(result, pair_div(residual, pair_mul_scalar(result, 2.0)));
    }
    return result;
}
fn pair_exp(a: vec2<f32>) -> vec2<f32> {
    let ln2 = vec2<f32>(0.693147182464599609375, -1.904654323148236e-9);
    let exponent = i32(round(a.x * 1.4426950408889634));
    let reduced = pair_add(a, -pair_mul_scalar(ln2, f32(exponent)));
    var term = vec2<f32>(1.0, 0.0);
    var result = term;
    for (var i = 1u; i <= 20u; i++) {
        term = pair_div(pair_mul(term, reduced), vec2<f32>(f32(i), 0.0));
        result = pair_add(result, term);
    }
    return pair_normalize(ldexp(result.x, exponent), ldexp(result.y, exponent));
}
fn pair_log(a: vec2<f32>) -> vec2<f32> {
    let decomposition = frexp(a.x);
    var exponent = decomposition.exp;
    var mantissa = vec2<f32>(decomposition.fract, ldexp(a.y, -exponent));
    if (mantissa.x < 0.7071067811865476) {
        mantissa *= 2.0;
        exponent -= 1;
    }
    let one = vec2<f32>(1.0, 0.0);
    let z = pair_div(pair_add(mantissa, -one), pair_add(mantissa, one));
    let square = pair_mul(z, z);
    var power = z;
    var series = z;
    for (var i = 3u; i <= 41u; i += 2u) {
        power = pair_mul(power, square);
        series = pair_add(series, pair_div(power, vec2<f32>(f32(i), 0.0)));
    }
    let ln2 = vec2<f32>(0.693147182464599609375, -1.904654323148236e-9);
    return pair_add(pair_mul_scalar(series, 2.0), pair_mul_scalar(ln2, f32(exponent)));
}
fn pair_sigmoid(a: vec2<f32>) -> vec2<f32> {
    let one = vec2<f32>(1.0, 0.0);
    let nonnegative = a.x >= 0.0;
    // Keep one exponential callsite while preserving each branch's exact input.
    let exponential = pair_exp(select(a, -a, nonnegative));
    if (nonnegative) { return pair_div(one, pair_add(one, exponential)); }
    return pair_div(exponential, pair_add(one, exponential));
}
fn pair_silu(a: vec2<f32>) -> vec2<f32> {
    return pair_mul(a, pair_sigmoid(a));
}
fn pair_softplus(a: vec2<f32>) -> vec2<f32> {
    let one = vec2<f32>(1.0, 0.0);
    if (a.x > 0.0) { return pair_add(a, pair_log(pair_add(one, pair_exp(-a)))); }
    return pair_log(pair_add(one, pair_exp(a)));
}
fn pair_sin_cos(a: vec2<f32>) -> vec4<f32> {
    let half_pi = vec2<f32>(1.57079637050628662109375, -4.37113900018624283e-8);
    let quadrant = i32(round(a.x * 0.63661977236758134));
    // Subtract the large high product before combining the small terms. Making
    // a normalized pair for the whole quadrant*pi/2 first would discard bits
    // that become significant after this cancellation.
    let high_product = half_pi.x * f32(quadrant);
    let high_product_low = product_residual(half_pi.x, f32(quadrant), high_product);
    let first_reduction = pair_add(a, -vec2<f32>(high_product, high_product_low));
    let tail = vec2<f32>(half_pi.y, -1.7151244994428829e-15);
    let reduced = pair_add(first_reduction, -pair_mul_scalar(tail, f32(quadrant)));
    let negative_square = -pair_mul(reduced, reduced);
    var sin_term = reduced;
    var cosine_term = vec2<f32>(1.0, 0.0);
    var sine = sin_term;
    var cosine = cosine_term;
    for (var i = 1u; i <= 12u; i++) {
        sin_term = pair_div(pair_mul(sin_term, negative_square), vec2<f32>(f32((2u*i)*(2u*i+1u)), 0.0));
        cosine_term = pair_div(pair_mul(cosine_term, negative_square), vec2<f32>(f32((2u*i-1u)*(2u*i)), 0.0));
        sine = pair_add(sine, sin_term);
        cosine = pair_add(cosine, cosine_term);
    }
    let sector = u32(quadrant) & 3u;
    if (sector == 1u) { return vec4<f32>(cosine, -sine); }
    if (sector == 2u) { return vec4<f32>(-sine, -cosine); }
    if (sector == 3u) { return vec4<f32>(-cosine, sine); }
    return vec4<f32>(sine, cosine);
}
