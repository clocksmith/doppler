import assert from 'node:assert/strict';

if (typeof globalThis.GPUBufferUsage === 'undefined') {
  globalThis.GPUBufferUsage = {
    MAP_READ: 0x0001,
    MAP_WRITE: 0x0002,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    INDEX: 0x0010,
    VERTEX: 0x0020,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
    INDIRECT: 0x0100,
    QUERY_RESOLVE: 0x0200,
  };
}

if (typeof globalThis.GPUMapMode === 'undefined') {
  globalThis.GPUMapMode = {
    READ: 0x0001,
    WRITE: 0x0002,
  };
}

if (typeof globalThis.GPUShaderStage === 'undefined') {
  globalThis.GPUShaderStage = {
    VERTEX: 0x1,
    FRAGMENT: 0x2,
    COMPUTE: 0x4,
  };
}

if (typeof globalThis.GPUTextureUsage === 'undefined') {
  globalThis.GPUTextureUsage = {
    COPY_SRC: 0x01,
    COPY_DST: 0x02,
    TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08,
    RENDER_ATTACHMENT: 0x10,
  };
}

const { runBrowserSuite, buildSuiteSummary } = await import('../../src/inference/browser-harness.js');
const { trainingHarness } = await import('../../src/training/suite.js');

{
  const startTime = performance.now() - 5;
  const summary = buildSuiteSummary('training', [
    { name: 'runner-smoke', passed: true, duration: 1 },
    { name: 'legacy-case', passed: true, skipped: true, duration: 1 },
  ], startTime);
  assert.equal(summary.suite, 'training');
  assert.equal(summary.passed, 1);
  assert.equal(summary.failed, 0);
  assert.equal(summary.skipped, 1);
  assert.ok(Number.isFinite(summary.duration));
}

{
  const tests = trainingHarness.listTests();
  assert.ok(Array.isArray(tests));
  assert.ok(tests.includes('runner-smoke'));
  assert.ok(tests.includes('train-step-metrics'));
  assert.ok(tests.includes('ul-stage1'));
  assert.ok(tests.includes('ul-stage2'));
}

await assert.rejects(
  () => runBrowserSuite({
    suite: 'unknown-suite',
    command: 'verify',
    surface: 'node',
  }),
  (error) => {
    assert.equal(error.code, 'unsupported_suite');
    assert.equal(error.requestedSuite, 'unknown-suite');
    assert.equal(error.command, 'verify');
    assert.equal(error.surface, 'node');
    assert.ok(Array.isArray(error.allowedSuites));
    assert.ok(error.allowedSuites.includes('training'));
    assert.ok(error.allowedSuites.includes('inference'));
    return true;
  }
);

await assert.rejects(
  () => runBrowserSuite({
    suite: undefined,
    command: 'verify',
    surface: 'node',
  }),
  (error) => {
    assert.equal(error.code, 'unsupported_suite');
    assert.equal(error.requestedSuite, '');
    return true;
  }
);

await assert.rejects(
  () => runBrowserSuite({
    suite: '',
    command: 'verify',
    surface: 'node',
  }),
  (error) => {
    assert.equal(error.code, 'unsupported_suite');
    assert.equal(error.requestedSuite, '');
    return true;
  }
);

console.log('browser-harness-suite-routing.test: ok');
