import assert from 'node:assert/strict';
import { runBrowserSuite } from '../../src/inference/browser-harness.js';
import { resetRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';
import { normalizeToolingCommandRequest } from '../../src/tooling/command-api.js';

const PROMPT = 'deterministic diffusion smoke prompt';
const NEGATIVE_PROMPT = 'low quality, noisy';
const WIDTH = 4;
const HEIGHT = 4;
const STEPS = 4;
const GUIDANCE = 6.5;
const WARMUP_RUNS = 1;
const TIMED_RUNS = 2;

const PIXELS = Uint8ClampedArray.from([
  22, 30, 41, 255, 28, 36, 48, 255, 36, 45, 58, 255, 45, 55, 69, 255,
  26, 34, 45, 255, 32, 41, 53, 255, 40, 50, 63, 255, 49, 60, 74, 255,
  31, 39, 50, 255, 37, 46, 57, 255, 45, 54, 67, 255, 54, 64, 78, 255,
  36, 44, 54, 255, 43, 51, 62, 255, 51, 60, 71, 255, 60, 70, 82, 255,
]);

const RUN_STATS = Object.freeze([
  Object.freeze({
    totalTimeMs: 88,
    prefillTimeMs: 18,
    decodeTimeMs: 70,
    vaeTimeMs: 8,
    prefillTokens: 32,
    decodeTokens: STEPS,
    gpu: {
      available: true,
      totalMs: 80,
      prefillMs: 16,
      denoiseMs: 64,
      vaeMs: 7,
    },
  }),
  Object.freeze({
    totalTimeMs: 90,
    prefillTimeMs: 20,
    decodeTimeMs: 70,
    vaeTimeMs: 10,
    prefillTokens: 32,
    decodeTokens: STEPS,
    gpu: {
      available: true,
      totalMs: 82,
      prefillMs: 18,
      denoiseMs: 64,
      vaeMs: 8,
    },
  }),
  Object.freeze({
    totalTimeMs: 98,
    prefillTimeMs: 24,
    decodeTimeMs: 74,
    vaeTimeMs: 12,
    prefillTokens: 32,
    decodeTokens: STEPS,
    gpu: {
      available: true,
      totalMs: 90,
      prefillMs: 21,
      denoiseMs: 69,
      vaeMs: 10,
    },
  }),
]);

const DIFFUSION_GEMMA_CONTRACT = Object.freeze({
  canvasLength: 4,
  maxDenoisingSteps: 2,
  maxNewTokens: 4,
  tMin: 0.4,
  tMax: 0.8,
  entropyBound: 0.1,
  confidenceThreshold: 0.005,
  stabilityThreshold: 1,
  padTokenId: 0,
  eosTokenIds: [1, 106, 50],
  boiTokenId: 255999,
  eoiTokenId: 258882,
  imageTokenId: 258880,
  selfConditioning: true,
  decoderCacheMode: 'encoder_kv_readonly_canvas_concat',
  router: {
    scaleHiddenStates: true,
    normalizeTopK: true,
    perExpertScale: true,
  },
});

const DIFFUSION_GEMMA_RUN_STATS = Object.freeze([
  Object.freeze({
    totalTimeMs: 36,
    prefillTimeMs: 4,
    decodeTimeMs: 32,
    prefillTokens: 2,
    decodeTokens: 4,
    denoiseSteps: 2,
    tokensPerForward: 2,
  }),
  Object.freeze({
    totalTimeMs: 40,
    prefillTimeMs: 5,
    decodeTimeMs: 35,
    prefillTokens: 2,
    decodeTokens: 4,
    denoiseSteps: 2,
    tokensPerForward: 2,
  }),
  Object.freeze({
    totalTimeMs: 44,
    prefillTimeMs: 6,
    decodeTimeMs: 38,
    prefillTokens: 2,
    decodeTokens: 4,
    denoiseSteps: 2,
    tokensPerForward: 2,
  }),
]);

function createDiffusionHarnessOverride() {
  let runIndex = 0;
  let stats = null;

  return {
    manifest: {
      modelId: 'sd3-diffusion-smoke',
      modelType: 'diffusion',
    },
    modelLoadMs: 7,
    pipeline: {
      reset() {},
      async generate(request) {
        assert.equal(request.prompt, PROMPT);
        assert.equal(request.negativePrompt, NEGATIVE_PROMPT);
        assert.equal(request.width, WIDTH);
        assert.equal(request.height, HEIGHT);
        assert.equal(request.steps, STEPS);
        assert.equal(request.guidanceScale, GUIDANCE);

        const sampleIndex = Math.min(runIndex, RUN_STATS.length - 1);
        stats = RUN_STATS[sampleIndex];
        runIndex += 1;

        return {
          width: WIDTH,
          height: HEIGHT,
          pixels: PIXELS,
        };
      },
      getStats() {
        return stats;
      },
      getMemoryStats() {
        return {
          used: 1024,
          kvCache: null,
        };
      },
      async unload() {
      },
    },
  };
}

function createDiffusionGemmaHarnessOverride() {
  let runIndex = 0;
  let stats = null;

  return {
    manifest: {
      modelId: 'diffusiongemma-26b-a4b-smoke',
      modelType: 'diffusion_gemma',
      architecture: {
        vocabSize: 262144,
      },
      inference: {
        ...DEFAULT_MANIFEST_INFERENCE,
        diffusionGemma: DIFFUSION_GEMMA_CONTRACT,
      },
    },
    modelLoadMs: 9,
    pipeline: {
      tokenizer: {
        decode(ids) {
          return ids.map((id) => `<${id}>`).join('');
        },
      },
      async generateTokenIds(prompt, options) {
        assert.equal(prompt, PROMPT);
        assert.equal(options.maxNewTokens, DIFFUSION_GEMMA_CONTRACT.maxNewTokens);
        assert.equal(options.seed, 1234);
        const sampleIndex = Math.min(runIndex, DIFFUSION_GEMMA_RUN_STATS.length - 1);
        stats = DIFFUSION_GEMMA_RUN_STATS[sampleIndex];
        runIndex += 1;
        return Int32Array.from([21, 22, 23, 24]);
      },
      async *generate() {
      },
      getStats() {
        return stats;
      },
      getMemoryStats() {
        return {
          used: 2048,
          kvCache: null,
        };
      },
      async unload() {
      },
    },
  };
}

function configureRuntime() {
  setRuntimeConfig({
    inference: {
      prompt: PROMPT,
      diffusion: {
        negativePrompt: NEGATIVE_PROMPT,
        scheduler: {
          numSteps: STEPS,
          guidanceScale: GUIDANCE,
        },
        latent: {
          width: WIDTH,
          height: HEIGHT,
        },
      },
    },
    shared: {
      benchmark: {
        run: {
          warmupRuns: WARMUP_RUNS,
          timedRuns: TIMED_RUNS,
        },
      },
    },
  });
}

function configureDiffusionGemmaRuntime() {
  setRuntimeConfig({
    inference: {
      prompt: PROMPT,
      diffusionGemma: {
        maxNewTokens: DIFFUSION_GEMMA_CONTRACT.maxNewTokens,
        seed: 1234,
      },
    },
    shared: {
      benchmark: {
        run: {
          warmupRuns: WARMUP_RUNS,
          timedRuns: TIMED_RUNS,
        },
      },
    },
  });
}

function assertDiffusionArtifacts(result, expectedSuite) {
  assert.equal(result.suite, expectedSuite);
  assert.equal(result.failed, 0);
  assert.ok(result.passed >= 1);
  assert.ok(result.metrics && typeof result.metrics === 'object');
  assert.equal(result.metrics.warmupRuns, WARMUP_RUNS);
  assert.equal(result.metrics.timedRuns, TIMED_RUNS);
  assert.equal(result.metrics.width, WIDTH);
  assert.equal(result.metrics.height, HEIGHT);
  assert.equal(result.metrics.steps, STEPS);
  assert.equal(result.metrics.guidanceScale, GUIDANCE);

  const perf = result.metrics.performanceArtifact;
  assert.ok(perf && typeof perf === 'object');
  assert.equal(perf.schemaVersion, 1);
  assert.equal(perf.warmupRuns, WARMUP_RUNS);
  assert.equal(perf.timedRuns, TIMED_RUNS);
  assert.ok(Number.isFinite(perf.cpu.prefillMs));
  assert.ok(Number.isFinite(perf.cpu.denoiseMs));
  assert.ok(Number.isFinite(perf.cpu.vaeMs));
  assert.ok(Number.isFinite(perf.cpu.totalMs));
  assert.ok(Number.isFinite(perf.throughput.prefillTokensPerSec));
  assert.ok(Number.isFinite(perf.throughput.decodeTokensPerSec));
  assert.ok(Number.isFinite(perf.throughput.decodeStepsPerSec));

  assert.ok(result.timingDiagnostics && typeof result.timingDiagnostics === 'object');
  assert.equal(result.timingDiagnostics.schemaVersion, 1);
  assert.equal(result.timingDiagnostics.source, 'doppler');
  assert.equal(result.timingDiagnostics.semantics.prefillMs, 'internal_prefill_phase');
}

function assertDiffusionGemmaArtifacts(result, expectedSuite) {
  assert.equal(result.suite, expectedSuite);
  assert.equal(result.failed, 0);
  assert.ok(result.passed >= 1);
  assert.ok(result.metrics && typeof result.metrics === 'object');
  assert.equal(result.metrics.warmupRuns, WARMUP_RUNS);
  assert.equal(result.metrics.timedRuns, TIMED_RUNS);
  assert.equal(result.metrics.canvasLength, DIFFUSION_GEMMA_CONTRACT.canvasLength);
  assert.equal(result.metrics.maxNewTokens, DIFFUSION_GEMMA_CONTRACT.maxNewTokens);
  assert.equal(result.metrics.steps, 2);
  assert.equal(result.metrics.guidanceScale, null);

  const perf = result.metrics.performanceArtifact;
  assert.ok(perf && typeof perf === 'object');
  assert.equal(perf.schemaVersion, 1);
  assert.equal(perf.modality, 'text');
  assert.equal(perf.shape.width, null);
  assert.equal(perf.shape.height, null);
  assert.equal(perf.scheduler.steps, 2);
  assert.equal(perf.scheduler.guidanceScale, null);
  assert.equal(perf.diffusionGemma.canvasLength, DIFFUSION_GEMMA_CONTRACT.canvasLength);
  assert.equal(perf.diffusionGemma.maxNewTokens, DIFFUSION_GEMMA_CONTRACT.maxNewTokens);
  assert.equal(perf.diffusionGemma.tokensPerForward.median, 2);
  assert.ok(Number.isFinite(perf.cpu.prefillMs));
  assert.ok(Number.isFinite(perf.cpu.denoiseMs));
  assert.equal(perf.cpu.vaeMs, 0);
  assert.ok(Number.isFinite(perf.throughput.decodeTokensPerSec));
  assert.ok(Number.isFinite(perf.throughput.decodeStepsPerSec));

  assert.ok(result.timingDiagnostics && typeof result.timingDiagnostics === 'object');
  assert.equal(result.timingDiagnostics.schemaVersion, 1);
  assert.equal(result.timingDiagnostics.source, 'doppler');
  assert.equal(result.timingDiagnostics.semantics.prefillMs, 'diffusion_gemma_encoder_prefill_phase');
  assert.equal(result.timingDiagnostics.semantics.decodeMs, 'diffusion_gemma_canvas_denoising_phase');
}

try {
  const verifyRequest = normalizeToolingCommandRequest({
    command: 'verify',
    workload: 'diffusion',
    modelId: 'sd3-diffusion-smoke',
  });
  assert.equal(verifyRequest.command, 'verify');
  assert.equal(verifyRequest.workload, 'diffusion');
  assert.equal(verifyRequest.modelId, 'sd3-diffusion-smoke');

  const benchRequest = normalizeToolingCommandRequest({
    command: 'bench',
    workload: 'diffusion',
    workloadType: 'diffusion',
    modelId: 'sd3-diffusion-smoke',
  });
  assert.equal(benchRequest.command, 'bench');
  assert.equal(benchRequest.workload, 'diffusion');
  assert.equal(benchRequest.workloadType, 'diffusion');

  configureRuntime();
  const verifyResult = await runBrowserSuite({
    command: 'verify',
    workload: 'diffusion',
    modelId: 'sd3-diffusion-smoke',
    cacheMode: 'warm',
    loadMode: 'memory',
    captureOutput: true,
    harnessOverride: createDiffusionHarnessOverride(),
  });

  assertDiffusionArtifacts(verifyResult, 'diffusion');
  assert.ok(!Object.hasOwn(verifyResult.metrics, 'workloadType'));
  assert.equal(verifyResult.output?.width, WIDTH);
  assert.equal(verifyResult.output?.height, HEIGHT);
  assert.equal(verifyResult.output?.pixels?.length, PIXELS.length);
  assert.equal(verifyResult.timing.cacheMode, 'warm');
  assert.equal(verifyResult.timing.loadMode, 'memory');

  configureRuntime();
  const benchResult = await runBrowserSuite({
    command: 'bench',
    workload: 'diffusion',
    workloadType: 'diffusion',
    modelId: 'sd3-diffusion-smoke',
    cacheMode: 'warm',
    loadMode: 'memory',
    captureOutput: true,
    harnessOverride: createDiffusionHarnessOverride(),
  });

  assertDiffusionArtifacts(benchResult, 'bench');
  assert.equal(benchResult.metrics.workloadType, 'diffusion');
  assert.equal(benchResult.output?.width, WIDTH);
  assert.equal(benchResult.output?.height, HEIGHT);
  assert.equal(benchResult.output?.pixels?.length, PIXELS.length);
  assert.equal(benchResult.timing.cacheMode, 'warm');
  assert.equal(benchResult.timing.loadMode, 'memory');

  const diffusionGemmaBenchRequest = normalizeToolingCommandRequest({
    command: 'bench',
    workload: 'diffusion',
    workloadType: 'diffusion_gemma',
    modelId: 'diffusiongemma-26b-a4b-smoke',
  });
  assert.equal(diffusionGemmaBenchRequest.command, 'bench');
  assert.equal(diffusionGemmaBenchRequest.workload, 'diffusion');
  assert.equal(diffusionGemmaBenchRequest.workloadType, 'diffusion_gemma');

  configureDiffusionGemmaRuntime();
  const diffusionGemmaVerifyResult = await runBrowserSuite({
    command: 'verify',
    workload: 'diffusion',
    modelId: 'diffusiongemma-26b-a4b-smoke',
    cacheMode: 'warm',
    loadMode: 'memory',
    captureOutput: true,
    harnessOverride: createDiffusionGemmaHarnessOverride(),
  });

  assertDiffusionGemmaArtifacts(diffusionGemmaVerifyResult, 'diffusion');
  assert.ok(!Object.hasOwn(diffusionGemmaVerifyResult.metrics, 'workloadType'));
  assert.deepEqual(diffusionGemmaVerifyResult.output?.tokenIds, [21, 22, 23, 24]);
  assert.equal(diffusionGemmaVerifyResult.output?.text, '<21><22><23><24>');
  assert.equal(diffusionGemmaVerifyResult.timing.cacheMode, 'warm');
  assert.equal(diffusionGemmaVerifyResult.timing.loadMode, 'memory');

  configureDiffusionGemmaRuntime();
  const diffusionGemmaBenchResult = await runBrowserSuite({
    command: 'bench',
    workload: 'diffusion',
    workloadType: 'diffusion_gemma',
    modelId: 'diffusiongemma-26b-a4b-smoke',
    cacheMode: 'warm',
    loadMode: 'memory',
    captureOutput: true,
    harnessOverride: createDiffusionGemmaHarnessOverride(),
  });

  assertDiffusionGemmaArtifacts(diffusionGemmaBenchResult, 'bench');
  assert.equal(diffusionGemmaBenchResult.metrics.workloadType, 'diffusion_gemma');
  assert.deepEqual(diffusionGemmaBenchResult.output?.tokenIds, [21, 22, 23, 24]);
  assert.equal(diffusionGemmaBenchResult.timing.cacheMode, 'warm');
  assert.equal(diffusionGemmaBenchResult.timing.loadMode, 'memory');
} finally {
  resetRuntimeConfig();
}

console.log('diffusion-command-smoke.test: ok');
