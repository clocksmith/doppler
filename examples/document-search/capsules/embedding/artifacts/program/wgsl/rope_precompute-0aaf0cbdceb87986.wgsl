override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    max_seq_len: u32,
    rotary_dim: u32,
    frequency_base_dim: u32,
    scaling_type: u32,
    theta: f32,
    rope_scale: f32,
    yarn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    original_max_position: f32,
    dispatch_stride: u32,
    mrope_section_t: u32,
    mrope_section_h: u32,
    mrope_section_w: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> rope_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> cos_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> sin_values: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_dim = u.rotary_dim / 2u;
    let index = gid.y * u.dispatch_stride + gid.x;
    let count = u.max_seq_len * half_dim;
    if (index >= count) {
        return;
    }
    let position = index / half_dim;
    let dimension = index % half_dim;
    let exponent = f32(dimension * 2u) / f32(u.frequency_base_dim);
    let frequency = 1.0 / pow(u.theta, exponent);
    var scale = u.rope_scale;
    var magnitude = 1.0;
    var rope_position = f32(position);
    if (u.scaling_type == 1u) {
        let wavelength = 2.0 * 3.141592653589793 / frequency;
        let low_threshold = u.original_max_position / u.yarn_beta_slow;
        let high_threshold = u.original_max_position / u.yarn_beta_fast;
        if (wavelength < high_threshold) {
            scale = 1.0;
        } else if (wavelength > low_threshold) {
            scale = u.yarn_factor;
        } else {
            let mix = (wavelength - high_threshold) / (low_threshold - high_threshold);
            scale = 1.0 + (u.yarn_factor - 1.0) * mix;
        }
    } else if (u.scaling_type == 2u) {
        scale = bitcast<f32>(rope_data[dimension]);
        magnitude = sqrt(
            1.0 + log(f32(u.max_seq_len) / u.original_max_position)
                / log(u.original_max_position)
        );
    } else if (u.scaling_type == 3u) {
        let temporal_end = u.mrope_section_t;
        let height_end = temporal_end + u.mrope_section_h;
        var axis = 2u;
        if (dimension < temporal_end) {
            axis = 0u;
        } else if (dimension < height_end) {
            axis = 1u;
        }
        rope_position = f32(rope_data[axis * u.max_seq_len + position]);
    }
    let angle = (rope_position / scale) * frequency;
    cos_values[index] = cos(angle) * magnitude;
    sin_values[index] = sin(angle) * magnitude;
}
