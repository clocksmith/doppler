import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';

// Identified, isolated arithmetic candidate; never an acceptance-policy change.
// Run only in the retained candidate source snapshot, not the baseline tree.
const file = new URL('../../src/gpu/kernels/attention_indexed.wgsl', import.meta.url);
const raw = await fs.readFile(file);
const hash = value => createHash('sha256').update(value).digest('hex');
// Raw file identity, distinct from the normalized runtime kernel-ref digest.
assert.equal(hash(raw), '49a68e1b3a635be70d123167a40c9452c7e7a228978931552be9644d1abd703d');
const anchor = `            var dot = 0.0;
            for (var d = 0u; d < head_dim; d = d + 1u) {
                dot = fma(shared_q[d], f32(K[k_base + d]), dot);
            }`;
const replacement = `            // Four independent accumulators shorten the dependent F32 sum.
            // Consecutive components retain their actual Q/K coordinates;
            // the final balanced sum improves the dot-product error bound.
            var sums = vec4<f32>(0.0);
            var d = 0u;
            for (; d + 3u < head_dim; d = d + 4u) {
                sums.x = fma(shared_q[d], K[k_base + d], sums.x);
                sums.y = fma(shared_q[d + 1u], K[k_base + d + 1u], sums.y);
                sums.z = fma(shared_q[d + 2u], K[k_base + d + 2u], sums.z);
                sums.w = fma(shared_q[d + 3u], K[k_base + d + 3u], sums.w);
            }
            var dot = (sums.x + sums.y) + (sums.z + sums.w);
            for (; d < head_dim; d = d + 1u) {
                dot = fma(shared_q[d], K[k_base + d], dot);
            }`;
const text = raw.toString('utf8');
assert.equal(text.split(anchor).length-1,1);
const candidate = text.replace(anchor,replacement);
await fs.writeFile(file,candidate);
console.log(JSON.stringify({ scope:'Isolated Q/K reduction candidate; same F32 KV, weights and selections algorithm',
  baselineSha256:hash(raw), candidateSha256:hash(candidate), generatorSha256:hash(await fs.readFile(new URL(import.meta.url))),
  qualified:false }));
