// One arithmetic experiment: unchanged Q4 decoding and F32 output storage.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const hash = value => createHash('sha256').update(value).digest('hex');
let source = await readFile('src/gpu/kernels/fused_matmul_q4_widetile.wgsl', 'utf8');
assert.equal(hash(source), 'd38300e9c30877e2aec62c5064ac8688cedd454146204d6cc1652a083ba0811c');
const helper = await readFile('src/gpu/kernels/codegen/includes/compensated-f32.wgsl.inc', 'utf8');
assert.equal(hash(helper), '1e877ff56eb6ef7e108ea1ce15c13bc9a82e1603e9372c50259c675fe1420ab5');
const replace = (from, to) => { assert.equal(source.split(from).length, 2, from); source = source.replace(from, to); };
replace('@compute @workgroup_size', helper + `
fn compensated_product(a: vec4<f32>, w: vec4<f32>, high: vec4<f32>, low: ptr<function, vec4<f32>>) -> vec4<f32> {
    var result: vec4<f32>;
    for (var lane = 0u; lane < 4u; lane++) {
        let product = a[lane] * w[lane];
        let next = pair_add(vec2<f32>(high[lane], (*low)[lane]),
                            vec2<f32>(product, product_residual(a[lane], w[lane], product)));
        result[lane] = next.x;
        (*low)[lane] = next.y;
    }
    return result;
}

@compute @workgroup_size`);
replace('var totals: array<f32, MAX_TILE_M>;', 'var totals: array<vec2<f32>, MAX_TILE_M>;');
replace('totals[i] = 0.0;', 'totals[i] = vec2<f32>(0.0);');
replace('var results: array<vec4<f32>, MAX_TILE_M>;', 'var results: array<vec4<f32>, MAX_TILE_M>;\n        var corrections: array<vec4<f32>, MAX_TILE_M>;');
replace('for (var m = 0u; m < TILE_M; m++) { results[m] = vec4<f32>(0.0); }',
  'for (var m = 0u; m < TILE_M; m++) { results[m] = vec4<f32>(0.0); corrections[m] = vec4<f32>(0.0); }');
replace('results[m] = fma(a_even, w_even, results[m]);',
  'results[m] = compensated_product(a_even, w_even, results[m], &corrections[m]);');
replace('results[m] = fma(a_odd, w_odd, results[m]);',
  'results[m] = compensated_product(a_odd, w_odd, results[m], &corrections[m]);');
replace(`var subtotal = 0.0;
                for (var lane = 0u; lane < 4u; lane++) { subtotal += results[m][lane]; }
                totals[m] += subtotal;`, `for (var lane = 0u; lane < 4u; lane++) {
                    totals[m] = pair_add(totals[m], vec2<f32>(results[m][lane], corrections[m][lane]));
                }`);
replace('C[row * u.N + col] = totals[m] * u.alpha;',
  'let scaled = pair_mul_scalar(totals[m], u.alpha);\n                C[row * u.N + col] = scaled.x + scaled.y;');
await writeFile('artifacts/recurrent-two-f32-20261012/q4-prefill-residual-candidate.wgsl', source);
console.log(JSON.stringify({ shaderSha256: hash(source), helperSha256: hash(helper),
  scope: 'Q4 prefill reduction only; unchanged encoded weights, materialized weights and F32 output storage' }));
