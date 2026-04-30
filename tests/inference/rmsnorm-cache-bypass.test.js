import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import {
  RMSNORM_CACHE_LIMIT,
  residualVariantBypassesCache,
} from '../../src/gpu/kernels/rmsnorm.js';

// The js-side limit must equal the WGSL-side MAX_CACHE_SIZE. If they drift,
// the cache-bypass predicate will permit hiddenSize values that silently OOB
// the shared_cache write, producing wrong output for wide-hidden models.
const wgslPath = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../src/gpu/kernels/rmsnorm.wgsl',
);
const wgslSource = readFileSync(wgslPath, 'utf8');
const match = wgslSource.match(/const\s+MAX_CACHE_SIZE\s*:\s*u32\s*=\s*(\d+)u/);
assert.ok(match, 'rmsnorm.wgsl must define MAX_CACHE_SIZE');
assert.equal(
  Number(match[1]),
  RMSNORM_CACHE_LIMIT,
  'RMSNORM_CACHE_LIMIT (js) must equal MAX_CACHE_SIZE (wgsl)',
);

assert.equal(residualVariantBypassesCache(null, 5376), false);
assert.equal(residualVariantBypassesCache(false, 5376), false);
assert.equal(residualVariantBypassesCache({}, null), false);
assert.equal(residualVariantBypassesCache({}, undefined), false);

assert.equal(residualVariantBypassesCache({}, RMSNORM_CACHE_LIMIT), false);
assert.equal(residualVariantBypassesCache({}, RMSNORM_CACHE_LIMIT + 1), true);

// Real-world hiddenSize values that must take the bypass.
assert.equal(residualVariantBypassesCache({}, 5120), true);
assert.equal(residualVariantBypassesCache({}, 5376), true); // Gemma 4 31B

// Real-world hiddenSize values that fit in cache and must keep the cached path.
assert.equal(residualVariantBypassesCache({}, 2048), false);
assert.equal(residualVariantBypassesCache({}, 3584), false);
assert.equal(residualVariantBypassesCache({}, 4096), false);

console.log('rmsnorm-cache-bypass.test: ok');
