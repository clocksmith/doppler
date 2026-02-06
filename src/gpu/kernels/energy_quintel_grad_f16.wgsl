// energy_quintel_grad_f16.wgsl
// Quintel gradient kernel for f16 state: compute dE/d(state) into f32 output.

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

const FLAG_MIRROR_X: u32 = 1u;
const FLAG_MIRROR_Y: u32 = 2u;
const FLAG_DIAGONAL: u32 = 4u;
const FLAG_COUNT: u32 = 8u;
const FLAG_CENTER: u32 = 16u;
const FLAG_BINARIZE: u32 = 32u;

struct Uniforms {
    count: u32,
    size: u32,
    flags: u32,
    _pad0: u32,
    stepSize: f32,
    gradientScale: f32,
    countDiff: f32,
    centerTarget: f32,
    symmetryWeight: f32,
    countWeight: f32,
    centerWeight: f32,
    binarizeWeight: f32,
    clampMin: f32,
    clampMax: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> state: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn hasFlag(mask: u32) -> bool {
    return (u.flags & mask) != 0u;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.count) {
        return;
    }

    let size = u.size;
    let i = idx / size;
    let j = idx - i * size;
    let value = f32(state[idx]);
    var grad: f32 = 0.0;

    if (hasFlag(FLAG_MIRROR_X)) {
        let mirrorJ = size - 1u - j;
        let mirrorIdx = i * size + mirrorJ;
        let diff = value - f32(state[mirrorIdx]);
        grad = grad + (2.0 * u.symmetryWeight * diff);
    }

    if (hasFlag(FLAG_MIRROR_Y)) {
        let mirrorI = size - 1u - i;
        let mirrorIdx = mirrorI * size + j;
        let diff = value - f32(state[mirrorIdx]);
        grad = grad + (2.0 * u.symmetryWeight * diff);
    }

    if (hasFlag(FLAG_DIAGONAL)) {
        let mirrorIdx = j * size + i;
        let diff = value - f32(state[mirrorIdx]);
        grad = grad + (2.0 * u.symmetryWeight * diff);
    }

    if (hasFlag(FLAG_COUNT)) {
        grad = grad + (2.0 * u.countWeight * u.countDiff);
    }

    if (hasFlag(FLAG_CENTER)) {
        let center = size / 2u;
        let centerIdx = center * size + center;
        if (idx == centerIdx) {
            let diff = value - u.centerTarget;
            grad = grad + (2.0 * u.centerWeight * diff);
        }
    }

    if (hasFlag(FLAG_BINARIZE)) {
        grad = grad + (u.binarizeWeight * (1.0 - 2.0 * value));
    }

    output[idx] = grad;
}

