import assert from 'node:assert/strict';

const { needsNormWeightOffset, isMoEModel } = await import('../../src/loader/manifest-config.js');

// === needsNormWeightOffset ===

// null/undefined manifest returns false
assert.equal(needsNormWeightOffset(null), false);
assert.equal(needsNormWeightOffset(undefined), false);

// rmsNormWeightOffset = true
{
  const manifest = {
    modelId: 'test-model',
    inference: { normalization: { rmsNormWeightOffset: true } },
  };
  assert.equal(needsNormWeightOffset(manifest), true);
}

// rmsNormWeightOffset = false
{
  const manifest = {
    modelId: 'test-model',
    inference: { normalization: { rmsNormWeightOffset: false } },
  };
  assert.equal(needsNormWeightOffset(manifest), false);
}

// Missing rmsNormWeightOffset field throws
{
  const manifest = {
    modelId: 'test-model',
    inference: { normalization: {} },
  };
  assert.throws(
    () => needsNormWeightOffset(manifest),
    /missing inference\.normalization\.rmsNormWeightOffset/
  );
}

// Missing normalization section throws
{
  const manifest = {
    modelId: 'test-model',
    inference: {},
  };
  assert.throws(
    () => needsNormWeightOffset(manifest),
    /missing inference\.normalization\.rmsNormWeightOffset/
  );
}

// Missing inference section throws
{
  const manifest = { modelId: 'test-model' };
  assert.throws(
    () => needsNormWeightOffset(manifest),
    /missing inference\.normalization\.rmsNormWeightOffset/
  );
}

// Error message includes modelId
{
  const manifest = { modelId: 'my-special-model', inference: {} };
  assert.throws(
    () => needsNormWeightOffset(manifest),
    /my-special-model/
  );
}

// Missing modelId uses 'unknown'
{
  const manifest = { inference: {} };
  assert.throws(
    () => needsNormWeightOffset(manifest),
    /unknown/
  );
}

// === isMoEModel ===

// null/undefined manifest returns false
assert.equal(isMoEModel(null), false);
assert.equal(isMoEModel(undefined), false);

// No moeConfig returns false
assert.equal(isMoEModel({}), false);
assert.equal(isMoEModel({ moeConfig: {} }), false);
assert.equal(isMoEModel({ moeConfig: { numExperts: 0 } }), false);
assert.equal(isMoEModel({ moeConfig: { numExperts: 1 } }), false);

// numExperts > 1 returns true
assert.equal(isMoEModel({ moeConfig: { numExperts: 2 } }), true);
assert.equal(isMoEModel({ moeConfig: { numExperts: 8 } }), true);
assert.equal(isMoEModel({ moeConfig: { numExperts: 64 } }), true);

console.log('manifest-config.test: ok');
