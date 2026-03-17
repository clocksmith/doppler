import assert from 'node:assert/strict';

import { DEFAULT_MOE_ROUTING_CONFIG, DEFAULT_MOE_CACHE_CONFIG } from '../../src/config/schema/moe.schema.js';

// === MoE routing defaults are sane for Gemma 4 ===

{
  assert.equal(DEFAULT_MOE_ROUTING_CONFIG.numExperts, 8);
  assert.equal(DEFAULT_MOE_ROUTING_CONFIG.topK, 2);
  assert.equal(DEFAULT_MOE_ROUTING_CONFIG.normalizeWeights, true);
  assert.equal(DEFAULT_MOE_ROUTING_CONFIG.routerDtype, 'f32');
  assert.ok(DEFAULT_MOE_ROUTING_CONFIG.maxTokensPerExpertHeadroom >= 1.0,
    'headroom must be >= 1.0');
}

// === MoE cache defaults are present ===

{
  assert.ok(typeof DEFAULT_MOE_CACHE_CONFIG.dequantCacheMaxEntries === 'number');
  assert.ok(DEFAULT_MOE_CACHE_CONFIG.dequantCacheMaxEntries > 0);
}

// === Router config validates topK < numExperts ===

{
  const config = { ...DEFAULT_MOE_ROUTING_CONFIG };
  assert.ok(config.topK <= config.numExperts,
    `topK (${config.topK}) must be <= numExperts (${config.numExperts})`);
  assert.ok(config.topK >= 1, 'topK must be >= 1');
}

// === Expert cache schema defaults are reasonable for MoE ===

{
  const { DEFAULT_EXPERT_CACHE_CONFIG } = await import('../../src/config/schema/loading.schema.js');

  assert.ok(DEFAULT_EXPERT_CACHE_CONFIG.defaultSizeBytes > 0,
    'expert cache default size must be > 0');
  assert.ok(DEFAULT_EXPERT_CACHE_CONFIG.maxBufferPercentage > 0 &&
    DEFAULT_EXPERT_CACHE_CONFIG.maxBufferPercentage <= 1.0,
    'maxBufferPercentage must be in (0, 1]');
  assert.ok(DEFAULT_EXPERT_CACHE_CONFIG.evictionHighWatermark > 0 &&
    DEFAULT_EXPERT_CACHE_CONFIG.evictionHighWatermark <= 1.0,
    'evictionHighWatermark must be in (0, 1]');
  assert.ok(
    DEFAULT_EXPERT_CACHE_CONFIG.emergencyTrimToRatio <
    DEFAULT_EXPERT_CACHE_CONFIG.evictionHighWatermark,
    'emergencyTrimToRatio must be < evictionHighWatermark'
  );
}

// === Buffer pool budget defaults support fail-closed enforcement ===

{
  const { DEFAULT_BUFFER_POOL_BUDGET_CONFIG } = await import('../../src/config/schema/buffer-pool.schema.js');

  assert.equal(DEFAULT_BUFFER_POOL_BUDGET_CONFIG.hardFailOnBudgetExceeded, true,
    'default budget must fail closed');
  assert.ok(DEFAULT_BUFFER_POOL_BUDGET_CONFIG.highWatermarkRatio > 0,
    'highWatermarkRatio must be > 0');
  assert.ok(
    DEFAULT_BUFFER_POOL_BUDGET_CONFIG.emergencyTrimTargetRatio <
    DEFAULT_BUFFER_POOL_BUDGET_CONFIG.highWatermarkRatio,
    'emergencyTrimTargetRatio must be < highWatermarkRatio'
  );
}

// === Gemma 4 MoE preset values are consistent with routing schema ===

{
  const { resolvePreset } = await import('../../src/config/loader.js');
  const preset = resolvePreset('gemma4');
  const moe = preset.inference.moe;

  // Preset topK must be valid
  assert.ok(moe.topK >= 1, 'preset topK must be >= 1');
  assert.ok(moe.topK <= moe.numExperts,
    `preset topK (${moe.topK}) must be <= numExperts (${moe.numExperts})`);

  // numSharedExperts must be < numExperts
  assert.ok(moe.numSharedExperts < moe.numExperts,
    `numSharedExperts (${moe.numSharedExperts}) must be < numExperts (${moe.numExperts})`);

  // routerDtype must be in supportedActivationDtypes or f32
  assert.ok(
    moe.routerDtype === 'f32' || moe.supportedActivationDtypes.includes(moe.routerDtype),
    `routerDtype "${moe.routerDtype}" must be f32 or in supportedActivationDtypes`
  );
}

console.log('moe-routing-contract.test: ok');
