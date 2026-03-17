import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const tiersDir = path.join(repoRoot, 'src/config/presets/runtime/tiers');

const TIER_FILES = [
  'gemma4-32gb.json',
  'gemma4-24gb.json',
  'gemma4-16gb.json',
];

const GB = 1024 * 1024 * 1024;
const MB = 1024 * 1024;

const tiers = {};
for (const file of TIER_FILES) {
  const filePath = path.join(tiersDir, file);
  tiers[file] = JSON.parse(await fs.readFile(filePath, 'utf8'));
}

// === All tier files load and have required metadata ===

for (const [file, tier] of Object.entries(tiers)) {
  assert.ok(tier.id, `${file}: must have id`);
  assert.ok(tier.name, `${file}: must have name`);
  assert.ok(tier.description, `${file}: must have description`);
  assert.ok(tier.intent, `${file}: must have intent`);
  assert.ok(tier.stability, `${file}: must have stability`);
  assert.ok(tier.owner, `${file}: must have owner`);
  assert.ok(tier.createdAtUtc, `${file}: must have createdAtUtc`);
  assert.equal(tier.extends, 'default', `${file}: must extend default`);
}

// === Each tier pins all required runtime fields ===

function assertTierPinsFields(file, tier) {
  const rt = tier.runtime;
  assert.ok(rt, `${file}: must have runtime`);

  // Buffer pool budget
  const budget = rt.shared?.bufferPool?.budget;
  assert.ok(budget, `${file}: must pin shared.bufferPool.budget`);
  assert.ok(typeof budget.maxTotalBytes === 'number' && budget.maxTotalBytes > 0,
    `${file}: budget.maxTotalBytes must be a positive number`);
  assert.equal(budget.hardFailOnBudgetExceeded, true,
    `${file}: budget.hardFailOnBudgetExceeded must be true`);

  // Expert cache
  const expertCache = rt.loading?.expertCache;
  assert.ok(expertCache, `${file}: must pin loading.expertCache`);
  assert.ok(typeof expertCache.defaultSizeBytes === 'number' && expertCache.defaultSizeBytes > 0,
    `${file}: expertCache.defaultSizeBytes must be a positive number`);

  // Prefetch
  const prefetch = rt.loading?.prefetch;
  assert.ok(prefetch, `${file}: must pin loading.prefetch`);
  assert.equal(prefetch.enabled, true, `${file}: prefetch must be enabled`);
  assert.ok(typeof prefetch.layersAhead === 'number',
    `${file}: prefetch.layersAhead must be a number`);

  // Memory management
  const memMgmt = rt.loading?.memoryManagement;
  assert.ok(memMgmt, `${file}: must pin loading.memoryManagement`);
  assert.ok(typeof memMgmt.flushIntervalLayers === 'number',
    `${file}: flushIntervalLayers must be a number`);

  // KV cache
  const kvcache = rt.inference?.kvcache;
  assert.ok(kvcache, `${file}: must pin inference.kvcache`);
  assert.ok(kvcache.layout, `${file}: must pin kvcache.layout`);
  assert.ok(typeof kvcache.maxSeqLen === 'number' && kvcache.maxSeqLen > 0,
    `${file}: kvcache.maxSeqLen must be a positive number`);
  assert.equal(kvcache.kvDtype, 'f16', `${file}: kvcache.kvDtype must be f16`);

  // MoE config
  const moe = rt.inference?.moe;
  assert.ok(moe, `${file}: must pin inference.moe`);
  assert.ok(moe.routing, `${file}: must pin moe.routing`);
  assert.ok(moe.cache, `${file}: must pin moe.cache`);
}

for (const [file, tier] of Object.entries(tiers)) {
  assertTierPinsFields(file, tier);
}

// === Budget ordering: 16gb < 24gb < 32gb ===

{
  const budget16 = tiers['gemma4-16gb.json'].runtime.shared.bufferPool.budget.maxTotalBytes;
  const budget24 = tiers['gemma4-24gb.json'].runtime.shared.bufferPool.budget.maxTotalBytes;
  const budget32 = tiers['gemma4-32gb.json'].runtime.shared.bufferPool.budget.maxTotalBytes;

  assert.ok(budget16 < budget24, `16gb budget (${budget16}) must be < 24gb budget (${budget24})`);
  assert.ok(budget24 < budget32, `24gb budget (${budget24}) must be < 32gb budget (${budget32})`);
}

// === Expert cache ordering: 16gb < 24gb < 32gb ===

{
  const cache16 = tiers['gemma4-16gb.json'].runtime.loading.expertCache.defaultSizeBytes;
  const cache24 = tiers['gemma4-24gb.json'].runtime.loading.expertCache.defaultSizeBytes;
  const cache32 = tiers['gemma4-32gb.json'].runtime.loading.expertCache.defaultSizeBytes;

  assert.ok(cache16 < cache24, `16gb expert cache (${cache16}) must be < 24gb (${cache24})`);
  assert.ok(cache24 < cache32, `24gb expert cache (${cache24}) must be < 32gb (${cache32})`);
}

// === KV cache maxSeqLen ordering: 16gb <= 24gb <= 32gb ===

{
  const seq16 = tiers['gemma4-16gb.json'].runtime.inference.kvcache.maxSeqLen;
  const seq24 = tiers['gemma4-24gb.json'].runtime.inference.kvcache.maxSeqLen;
  const seq32 = tiers['gemma4-32gb.json'].runtime.inference.kvcache.maxSeqLen;

  assert.ok(seq16 <= seq24, `16gb maxSeqLen (${seq16}) must be <= 24gb (${seq24})`);
  assert.ok(seq24 <= seq32, `24gb maxSeqLen (${seq24}) must be <= 32gb (${seq32})`);
}

// === 16gb tier is explicitly constrained ===

{
  const tier16 = tiers['gemma4-16gb.json'];
  const budget = tier16.runtime.shared.bufferPool.budget;
  const expertCache = tier16.runtime.loading.expertCache;
  const kvcache = tier16.runtime.inference.kvcache;

  // Hard budget enforcement
  assert.equal(budget.hardFailOnBudgetExceeded, true,
    '16gb tier must fail closed on budget exceeded');

  // Budget must be < 16 GB (leave headroom for OS/browser)
  assert.ok(budget.maxTotalBytes < 16 * GB,
    `16gb tier budget (${budget.maxTotalBytes}) must be < 16 GB`);

  // Expert cache must be <= 1 GB
  assert.ok(expertCache.defaultSizeBytes <= 1 * GB,
    `16gb tier expert cache (${expertCache.defaultSizeBytes}) must be <= 1 GB`);

  // Short context
  assert.ok(kvcache.maxSeqLen <= 2048,
    `16gb tier maxSeqLen (${kvcache.maxSeqLen}) must be <= 2048`);

  // Minimal dequant cache
  assert.ok(tier16.runtime.inference.moe.cache.dequantCacheMaxEntries <= 4,
    '16gb tier dequant cache must be <= 4 entries');
}

// === 32gb tier has generous allocations ===

{
  const tier32 = tiers['gemma4-32gb.json'];
  const budget = tier32.runtime.shared.bufferPool.budget;
  const expertCache = tier32.runtime.loading.expertCache;
  const kvcache = tier32.runtime.inference.kvcache;

  // Budget must be > 24 GB
  assert.ok(budget.maxTotalBytes > 24 * GB,
    `32gb tier budget (${budget.maxTotalBytes}) must be > 24 GB`);

  // Expert cache >= 4 GB
  assert.ok(expertCache.defaultSizeBytes >= 4 * GB,
    `32gb tier expert cache (${expertCache.defaultSizeBytes}) must be >= 4 GB`);

  // Full context
  assert.ok(kvcache.maxSeqLen >= 8192,
    `32gb tier maxSeqLen (${kvcache.maxSeqLen}) must be >= 8192`);
}

// === All tiers pin hard budget enforcement ===

for (const [file, tier] of Object.entries(tiers)) {
  assert.equal(
    tier.runtime.shared.bufferPool.budget.hardFailOnBudgetExceeded,
    true,
    `${file}: all tiers must enforce hard budget`
  );
}

console.log('gemma4-tier-presets-contract.test: ok');
