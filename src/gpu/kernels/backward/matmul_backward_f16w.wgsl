// Matrix Multiplication Backward Kernel - dX = dY @ W^T with packed F16 W.
//
// dY and dX remain F32. W is stored as consecutive IEEE-754 binary16 values,
// packed two values per u32 storage word. unpack2x16float keeps this path
// available without requiring native shader-f16 arithmetic.

const MAX_TILE_SIZE: u32 = 16u;
const TILE_AREA: u32 = MAX_TILE_SIZE * MAX_TILE_SIZE;

override TILE_SIZE: u32 = 16u;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    transpose_b: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> dX: array<f32>;

var<workgroup> tileDY: array<f32, TILE_AREA>;
var<workgroup> tileW: array<f32, TILE_AREA>;

fn load_f16(index: u32) -> f32 {
    let pair = unpack2x16float(W[index >> 1u]);
    return select(pair.x, pair.y, (index & 1u) == 1u);
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    let local_row = local_id.x;
    let local_col = local_id.y;

    var sum: f32 = 0.0;
    let num_tiles = (u.N + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let tile_n = t * TILE_SIZE;
        let tile_idx = local_row * TILE_SIZE + local_col;
        let dy_col = tile_n + local_col;
        if (row < u.M && dy_col < u.N) {
            tileDY[tile_idx] = dY[row * u.N + dy_col];
        } else {
            tileDY[tile_idx] = 0.0;
        }

        let wt_row = tile_n + local_row;
        if (wt_row < u.N && col < u.K) {
            if (u.transpose_b == 0u) {
                tileW[tile_idx] = load_f16(col * u.N + wt_row);
            } else {
                tileW[tile_idx] = load_f16(wt_row * u.K + col);
            }
        } else {
            tileW[tile_idx] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum += tileDY[local_row * TILE_SIZE + k] * tileW[k * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    if (row < u.M && col < u.K) {
        dX[row * u.K + col] = sum * u.alpha;
    }
}
