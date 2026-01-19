// adam.wgsl

/**
 * Adam optimizer kernel.
 */

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    size: u32,
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> params: array<f32>;
@group(0) @binding(2) var<storage, read> grads: array<f32>;
@group(0) @binding(3) var<storage, read_write> moment1: array<f32>;
@group(0) @binding(4) var<storage, read_write> moment2: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.size) {
        return;
    }

    let g = grads[idx];
    let m = moment1[idx];
    let v = moment2[idx];

    let beta1 = u.beta1;
    let beta2 = u.beta2;

    m = beta1 * m + (1.0 - beta1) * g;
    v = beta2 * v + (1.0 - beta2) * g * g;

    let step_f = f32(u.step);
    let m_hat = m / (1.0 - pow(beta1, step_f));
    let v_hat = v / (1.0 - pow(beta2, step_f));

    params[idx] = params[idx] - u.lr * m_hat / (sqrt(v_hat) + u.eps);
    moment1[idx] = m;
    moment2[idx] = v;
}
