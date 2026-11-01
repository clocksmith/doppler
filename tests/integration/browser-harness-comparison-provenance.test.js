import assert from 'node:assert/strict';
import { runBrowserSuite } from '../../src/tooling/browser-harness.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';
import { buildComparisonFingerprint } from '../../src/client/inspection.js';
import { compareBenchResults } from '../../src/cli/cli-output.js';
import { sha256Hex } from '../../src/formats/sha256.js';

// CPU-only fixtures establish evidence plumbing, not GPU performance.
const originalConfig = structuredClone(getRuntimeConfig());
const manifest = {
  modelId: 'comparison-fixture', modelType: 'transformer',
  tokenizer: { type: 'fixture', vocabSize: 16 },
  architecture: { numLayers: 1, hiddenSize: 8, intermediateSize: 16,
    numAttentionHeads: 1, numKeyValueHeads: 1, headDim: 8, vocabSize: 16, maxSeqLen: 64 },
  inference: { attention: { queryPreAttnScalar: 8 }, chatTemplate: { type: 'none', enabled: false } },
};
const manifestText = JSON.stringify(manifest, null, 2);
const capabilities = { hasF16: true, hasSubgroups: false,
  adapterInfo: { vendor: 'fixture', architecture: 'cpu-test', device: 'mock', description: 'No GPU' } };

const digest = (character) => `sha256:${character.repeat(64)}`;
const resolvedRuntimeSession = {
  schema: 'doppler.resolved-runtime-session/v1',
  id: digest('9'),
  manifestInference: {
    execution: {
      kernels: {
        main: { kernel: 'main.wgsl', entry: 'main', digest: digest('6') },
        recurrent: { kernel: 'gated_delta_recurrent.wgsl', entry: 'main', digest: digest('a') },
      },
      mechanismKernels: ['recurrent'],
      preLayer: [['embed', 'main']], prefill: [['attention', 'main']],
      decode: [['attention', 'main']], postLayer: [['sample', 'main']],
    },
  },
  runtime: {
    session: {
      kvcache: { layout: 'contiguous', kvDtype: 'f32' },
      perLayerInputs: { materialization: 'eager' },
      largeWeights: { residency: 'gpu' },
    },
    compute: { outputDtype: 'f32' },
  },
  execution: {
    primary: { id: 'primary', activationDtype: 'f32' },
    resolvedSteps: {
      prefill: [{ id: 'prefill.attention' }],
      decode: [{ id: 'decode.attention' }],
      all: [{ id: 'prefill.attention' }, { id: 'decode.attention' }],
    },
    resolvedStepsHash: digest('7'),
    appliedTransforms: [{ id: 'fuse-qkv' }],
  },
  dtypes: { activation: 'f32', output: 'f32', kv: 'f32', math: 'f32', accumulation: 'f32' },
  kernelPath: { id: 'portable', source: 'capsule', hash: digest('8'), definition: {} },
  capabilityPolicy: { f16: false },
  laneIntegrity: { status: 'passed' },
};

async function runFixture({ missingHash = false, missingTokenizer = false, changedPrompt = false,
  changedPlan = false, changedRuntime = false, missingRuntime = false, instrumented = false, seed = null, missingAdapter = false } = {}) {
  let invocation = 0;
  let unloaded = false;
  const pipeline = {
    resolvedRuntimeSession: missingRuntime ? null : structuredClone(resolvedRuntimeSession),
    tokenizer: missingTokenizer ? null : {
      encode: () => changedPrompt && invocation > 0 ? [7, 9] : [7, 8],
      decode: () => 'answer',
    },
    reset() {
      if (changedRuntime && invocation > 0) this.resolvedRuntimeSession.runtime.session.kvcache.kvDtype = 'f16';
    },
    async unload() { unloaded = true; },
    async *generate(_prompt, options) {
      invocation += 1;
      options.onToken(1, 'answer');
      yield 'answer';
      options.onToken(2, '.');
      yield '.';
    },
    getStats() {
      assert.equal(unloaded, false, 'execution identity must be captured before unloading');
      const id = changedPlan && invocation > 1 ? 'other-plan' : 'fixture-plan';
      return { prefillTokens: 2, prefillTimeMs: 1, ttftMs: 1,
        decodeTimeMs: 1, decodeTokens: 1, totalTimeMs: 2, decodeProfileSteps: [],
        executionPlan: { finalActivePlanId: id,
          primary: { id, kernelPathId: 'fixture-kernel', kernelPathSource: 'manifest', activationDtype: 'f16' } } };
    },
  };
  setRuntimeConfig({
    inference: { prompt: 'fixture prompt', generation: { maxTokens: 2 },
      sampling: { temperature: 0, topK: 1, topP: 1, seed: null } },
    shared: { benchmark: { run: { warmupRuns: 0, timedRuns: 2, maxNewTokens: 2, seed: null,
      executionObserver: { enabled: instrumented } } }, debug: { profiler: { enabled: false }, probes: [] } },
  });
  const result = await runBrowserSuite({
    suite: 'bench', command: 'bench', workload: 'inference', surface: 'node', modelId: 'request-alias',
    ...(seed === null ? {} : { inferenceInput: { seed } }),
    harnessOverride: { modelLoadMs: 7, pipeline, manifest,
      capabilities: missingAdapter ? {} : capabilities,
      manifestHash: missingHash ? null : sha256Hex(manifestText) },
  });
  assert.equal(unloaded, true);
  return result;
}

try {
  const first = await runFixture();
  const second = await runFixture();
  assert.equal(first.modelId, manifest.modelId);
  assert.equal(first.workload, 'inference');
  assert.equal(first.env.runtime, 'node');
  assert.equal(first.metrics.seed, null);
  assert.equal(first.comparisonFingerprintUnavailableReason, null);
  const identity = first.comparisonFingerprint.identity;
  assert.equal(identity.artifact.manifestHash, `sha256:${sha256Hex(manifestText)}`);
  assert.deepEqual(identity.promptTokenIds, [7, 8]);
  assert.deepEqual(identity.tokenizer.contract, manifest.tokenizer);
  assert.match(identity.execution.initialExecutionIdentityDigest, /^sha256:[0-9a-f]{64}$/);
  assert.equal(identity.execution.executionPlanId, 'fixture-plan');
  assert.equal(identity.execution.activationDtype, 'f16');
  assert.deepEqual(identity.adapter, capabilities.adapterInfo);
  assert.equal(identity.sampling.maxTokens, 2);
  assert.equal(identity.sampling.seed, null);
  assert.deepEqual(first.comparisonFingerprint, buildComparisonFingerprint({
    artifact: identity.artifact, tokenizer: manifest.tokenizer, promptTokenIds: [7, 8],
    sampling: identity.sampling, observationPolicyId: identity.observationPolicy.id,
    execution: identity.execution, browser: identity.browser, adapter: identity.adapter,
  }));
  assert.doesNotThrow(() => compareBenchResults(first, second));
  const seeded = await runFixture({ seed: 31 });
  assert.equal(seeded.metrics.seed, 31);
  assert.equal(seeded.comparisonFingerprint.identity.sampling.seed, 31);
  assert.throws(() => compareBenchResults(first, seeded), /comparison|fingerprint/i);
  for (const options of [
    { missingHash: true }, { missingTokenizer: true }, { changedPrompt: true },
    { changedPlan: true }, { changedRuntime: true }, { missingRuntime: true },
    { instrumented: true }, { missingAdapter: true },
  ]) {
    const unavailable = await runFixture(options);
    assert.equal(unavailable.comparisonFingerprint, null, JSON.stringify(options));
    assert.ok(unavailable.comparisonFingerprintUnavailableReason);
    assert.throws(() => compareBenchResults(first, unavailable), /comparison|fingerprint/i);
  }
} finally {
  setRuntimeConfig(originalConfig);
}
console.log('browser-harness-comparison-provenance.test: ok (CPU fixtures; no GPU evidence)');
