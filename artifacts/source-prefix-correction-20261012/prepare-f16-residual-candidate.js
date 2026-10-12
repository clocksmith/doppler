// Isolated same-storage F32 accumulation experiment; no canonical shader edit.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const source = await readFile(new URL('../../src/gpu/kernels/matmul_f16w_f32a_tiled.wgsl', import.meta.url), 'utf8');
assert.equal(createHash('sha256').update(source).digest('hex'), '31b2f8d9623d77df223f12da61eb5a178209fa1ce43199fe94357d7791c2b3b5');
const norm = await readFile(new URL('./candidate-range-product-rmsnorm.wgsl', import.meta.url), 'utf8');
const helpers = norm.slice(norm.indexOf('fn add_u64'), norm.indexOf('var<workgroup> shared_error'));
const pairs = Array.from({ length: 16 }, (_, n) => [Math.floor(n / 4), n % 4]);
let candidate = source.replace('@compute', helpers + '\n@compute');
candidate = candidate.replace('    let num_tiles', pairs.map(([r, c]) => `    var error${r}${c}: f32 = 0.0;`).join('\n') + '\n\n    let num_tiles');
const begin = candidate.indexOf('        // Bound each local accumulation');
const end = candidate.indexOf('        // Compute:', begin);
assert(begin > 0 && end > begin);
candidate = candidate.slice(0, begin) + candidate.slice(end);
for (const [r, c] of pairs) {
  candidate = candidate.replace(`tile${r}${c} = fma(a${r}, b${c}, tile${r}${c});`, `{
                let product = a${r} * b${c};
                let total = acc${r}${c} + product;
                error${r}${c} += product_residual(a${r}, b${c}, product) + addition_residual(acc${r}${c}, product, total);
                acc${r}${c} = total;
            }`);
  candidate = candidate.replace(`acc${r}${c} += tile${r}${c};`, '');
  candidate = candidate.replace(`= acc${r}${c} * alpha;`, `= (acc${r}${c} + error${r}${c}) * alpha;`);
}
assert(!/tile[0-3][0-3]/.test(candidate));
await writeFile(new URL('./candidate-f16-residual-tiled.wgsl', import.meta.url), candidate);
