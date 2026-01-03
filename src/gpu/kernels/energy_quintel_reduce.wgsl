// energy_quintel_reduce.wgsl
// Reduces quintel energy components and state sum per workgroup.

const WORKGROUP_SIZE: u32 = 256u;

const FLAG_MIRROR_X: u32 = 1u;
const FLAG_MIRROR_Y: u32 = 2u;
const FLAG_DIAGONAL: u32 = 4u;
const FLAG_CENTER: u32 = 16u;
const FLAG_BINARIZE: u32 = 32u;

struct Uniforms {
    count: u32,
    size: u32,
    flags: u32,
    _pad0: u32,
    symmetryWeight: f32,
    centerWeight: f32,
    binarizeWeight: f32,
    centerTarget: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> state: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

var<workgroup> sharedAccum: array<vec4<f32>, WORKGROUP_SIZE>;

fn hasFlag(mask: u32) -> bool {
    return (u.flags & mask) != 0u;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;
    var accum = vec4<f32>(0.0);

    if (idx < u.count) {
        let size = u.size;
        let i = idx / size;
        let j = idx - i * size;
        let value = state[idx];

        accum.x = value;

        var symmetryEnergy: f32 = 0.0;
        if (hasFlag(FLAG_MIRROR_X)) {
            let mirrorJ = size - 1u - j;
            if (j < mirrorJ) {
                let mirrorIdx = i * size + mirrorJ;
                let diff = value - state[mirrorIdx];
                symmetryEnergy += u.symmetryWeight * diff * diff;
            }
        }
        if (hasFlag(FLAG_MIRROR_Y)) {
            let mirrorI = size - 1u - i;
            if (i < mirrorI) {
                let mirrorIdx = mirrorI * size + j;
                let diff = value - state[mirrorIdx];
                symmetryEnergy += u.symmetryWeight * diff * diff;
            }
        }
        if (hasFlag(FLAG_DIAGONAL)) {
            if (i < j) {
                let mirrorIdx = j * size + i;
                let diff = value - state[mirrorIdx];
                symmetryEnergy += u.symmetryWeight * diff * diff;
            }
        }
        accum.y = symmetryEnergy;

        if (hasFlag(FLAG_BINARIZE)) {
            accum.z = u.binarizeWeight * value * (1.0 - value);
        }

        if (hasFlag(FLAG_CENTER)) {
            let center = size / 2u;
            let centerIdx = center * size + center;
            if (idx == centerIdx) {
                let diff = value - u.centerTarget;
                accum.w = u.centerWeight * diff * diff;
            }
        }
    }

    sharedAccum[lid.x] = accum;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid.x < stride) {
            sharedAccum[lid.x] = sharedAccum[lid.x] + sharedAccum[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid.x == 0u) {
        output[wid.x] = sharedAccum[0];
    }
}
