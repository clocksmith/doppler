override ELEMENTS: u32;
var<workgroup> scratch: array<f32, ELEMENTS>;
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(ELEMENTS)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    scratch[local.x] = 1.0;
    workgroupBarrier();
    if (local.x == 0u) {
        var sum = 0.0;
        for (var index = 0u; index < ELEMENTS; index += 1u) {
            sum += scratch[index];
        }
        output[0] = sum;
    }
}
