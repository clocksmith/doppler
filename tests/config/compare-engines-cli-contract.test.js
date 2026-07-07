import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { mergeKernelPathPolicy } from '../../src/config/merge-helpers.js';
import {
  buildCompareSection,
  buildCompareFairnessAudit,
  buildDiagnosticSharedBenchmarkContract,
  buildDopplerBottleneckDiagnostic,
  buildDopplerManifestFreshnessArtifact,
  buildDopplerBottleneckDiagnosticRuntimeOverlay,
  buildDopplerRuntimeConfig,
  buildSharedBenchmarkContract,
  assertDopplerDecodeCadence,
  loadModelCatalogBundle,
  normalizeCompareLoadModeDefaults,
  parseArgs as parseCompareArgs,
  parseJsonBlock,
  parseOnOff as parseCompareOnOff,
  redactSecrets,
  resolveComputeDecodeCadence,
  usage as renderCompareUsage,
  renderComparePrompt,
  resolveCompareOwnedPromptRenderer,
  resolveCatalogTransformersjsBenchmarkTarget,
  resolveCompareProfile,
  resolveCompareLoadModes,
  resolveDopplerBenchmarkKvCachePlan,
  resolveDopplerExecutionIdentity,
  resolveDopplerModelSource,
  resolveDopplerThroughputCadenceGate,
  summarizeDopplerDecodeProfileSteps,
} from '../../tools/compare-engines.js';
import { buildTokenAccurateSyntheticPrompt } from '../../benchmarks/vendors/workload-prompt.js';

function runCompareEngines(args) {
  return spawnSync(process.execPath, ['tools/compare-engines.js', ...args], {
    cwd: process.cwd(),
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

function assertCommandOutputMatches(result, pattern) {
  const output = [result.stderr, result.stdout].filter(Boolean).join('\n');
  if (output.length === 0) {
    assert.notEqual(result.status, 0);
    return;
  }
  assert.match(output, pattern);
}

{
  const repoRoot = process.cwd();
  const compareConfigPath = path.join(repoRoot, 'benchmarks', 'vendors', 'compare-engines.config.json');
  const benchmarkPolicyPath = path.join(repoRoot, 'benchmarks', 'vendors', 'benchmark-policy.json');
  const catalogPath = path.join(repoRoot, 'models', 'catalog.json');
  const quickstartRegistryPath = path.join(repoRoot, 'src', 'client', 'doppler-registry.json');
  const compareConfig = JSON.parse(await fs.readFile(compareConfigPath, 'utf8'));
  const benchmarkPolicy = JSON.parse(await fs.readFile(benchmarkPolicyPath, 'utf8'));
  const catalog = JSON.parse(await fs.readFile(catalogPath, 'utf8'));
  const quickstartRegistry = JSON.parse(await fs.readFile(quickstartRegistryPath, 'utf8'));
  const catalogIds = new Set((Array.isArray(catalog.models) ? catalog.models : []).map((entry) => entry?.modelId).filter(Boolean));
  const quickstartIds = new Set((Array.isArray(quickstartRegistry.models) ? quickstartRegistry.models : []).map((entry) => entry?.modelId).filter(Boolean));
  const compareProfileIds = new Set(
    (Array.isArray(compareConfig.modelProfiles) ? compareConfig.modelProfiles : [])
      .map((entry) => entry?.dopplerModelId)
      .filter(Boolean)
  );
  const gemma3Profile = compareConfig.modelProfiles.find((entry) => entry?.dopplerModelId === 'gemma-3-270m-it-q4k-ehf16-af32') || null;
  const gemma3Q4kThroughputProfile = JSON.parse(await fs.readFile(
    path.join(repoRoot, 'src', 'config', 'runtime', 'profiles', 'gemma3-270m-q4k-throughput-overlapped-probe.json'),
    'utf8'
  ));
  const qwen08Profile = compareConfig.modelProfiles.find((entry) => entry?.dopplerModelId === 'qwen-3-5-0-8b-q4k-ehaf16') || null;
  const qwen2Profile = compareConfig.modelProfiles.find((entry) => entry?.dopplerModelId === 'qwen-3-5-2b-q4k-ehaf16') || null;
  const gemma4Profile = compareConfig.modelProfiles.find((entry) => entry?.dopplerModelId === 'gemma-4-e2b-it-q4k-ehf16-af32') || null;
  const gemma4Int4PleProfile = compareConfig.modelProfiles.find((entry) => entry?.dopplerModelId === 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple') || null;
  const gemma4Af16Int4PleProfile = compareConfig.modelProfiles.find((entry) => entry?.dopplerModelId === 'gemma-4-e2b-it-q4k-ehf16-af16-int4ple') || null;
  const gemma4Catalog = (Array.isArray(catalog.models) ? catalog.models : [])
    .find((entry) => entry?.modelId === 'gemma-4-e2b-it-q4k-ehf16-af32') || null;

  assert.deepEqual(compareConfig.defaults, {
    warmLoadMode: 'opfs',
    coldLoadMode: 'http',
  });
  {
    const parityRuntimeBaseConfig = {
      profile: 'parity-runtime',
      inference: {
        batching: {
          readbackMode: 'sequential',
        },
        session: {
          decodeLoop: {
            readbackMode: 'sequential',
          },
        },
      },
    };
    const customRuntimeBaseConfig = {
      profile: 'custom-runtime',
      inference: {
        batching: {
          readbackMode: 'overlapped',
        },
        session: {
          decodeLoop: {
            readbackMode: 'overlapped',
          },
        },
      },
    };
    const customCadence = {
      batchSize: 8,
      readbackInterval: 6,
      stopCheckMode: 'batch',
      readbackMode: 'overlapped',
      disableMultiTokenDecode: false,
      speculationMode: null,
    };
    assert.deepEqual(
      resolveComputeDecodeCadence('custom', 'parity', customCadence, customRuntimeBaseConfig),
      {
        batchSize: 8,
        readbackInterval: 6,
        stopCheckMode: 'batch',
        readbackMode: 'overlapped',
        disableMultiTokenDecode: false,
        speculationMode: null,
        runtimeBaseConfig: customRuntimeBaseConfig,
      }
    );
    assert.deepEqual(
      resolveComputeDecodeCadence('parity', 'parity', customCadence, parityRuntimeBaseConfig),
      {
        batchSize: benchmarkPolicy.decodeProfiles.profiles.parity.batchSize,
        readbackInterval: benchmarkPolicy.decodeProfiles.profiles.parity.readbackInterval,
        stopCheckMode: benchmarkPolicy.decodeProfiles.profiles.parity.stopCheckMode,
        readbackMode: 'sequential',
        disableMultiTokenDecode: benchmarkPolicy.decodeProfiles.profiles.parity.disableMultiTokenDecode === true,
        speculationMode: benchmarkPolicy.decodeProfiles.profiles.parity.speculationMode ?? null,
        runtimeBaseConfig: parityRuntimeBaseConfig,
      }
    );
    const matchingDopplerResult = {
      request: {
        runtimeConfig: {
          inference: {
            generation: {
              disableMultiTokenDecode: false,
            },
            batching: {
              batchSize: 8,
              readbackInterval: 6,
              stopCheckMode: 'batch',
              readbackMode: 'overlapped',
            },
            session: {
              decodeLoop: {
                batchSize: 8,
                readbackInterval: 6,
                stopCheckMode: 'batch',
                readbackMode: 'overlapped',
                disableCommandBatching: false,
              },
            },
          },
        },
      },
    };
    assert.doesNotThrow(() => {
      assertDopplerDecodeCadence(matchingDopplerResult, customCadence, 'unit/custom');
    });
    const cadenceSection = buildCompareSection({
      cacheMode: 'warm',
      loadMode: 'opfs',
      doppler: matchingDopplerResult,
      transformersjs: { runs: [] },
    });
    assert.deepEqual(cadenceSection.dopplerDecodeCadence, {
      batchSize: 8,
      readbackInterval: 6,
      stopCheckMode: 'batch',
      readbackMode: 'overlapped',
      disableCommandBatching: false,
      disableMultiTokenDecode: false,
      speculationMode: null,
      tokensPerReadback: 48,
      runtimeMirror: {
        batching: {
          batchSize: 8,
          readbackInterval: 6,
          stopCheckMode: 'batch',
          readbackMode: 'overlapped',
        },
        decodeLoop: {
          batchSize: 8,
          readbackInterval: 6,
          stopCheckMode: 'batch',
          readbackMode: 'overlapped',
        },
      },
    });
    const measuredCadenceSection = buildCompareSection({
      cacheMode: 'warm',
      loadMode: 'opfs',
      doppler: {
        request: {
          runtimeConfig: {
            inference: {
              generation: {
                disableMultiTokenDecode: false,
              },
              batching: {
                batchSize: 4,
                readbackInterval: 4,
                stopCheckMode: 'batch',
              },
              session: {
                decodeLoop: {
                  batchSize: 4,
                  readbackInterval: 4,
                  stopCheckMode: 'batch',
                  disableCommandBatching: false,
                },
              },
            },
          },
        },
        result: {
          metrics: {
            decodeCadence: {
              batchSize: 4,
              readbackInterval: 4,
              stopCheckMode: 'batch',
              readbackMode: 'sequential',
              disableCommandBatching: false,
              disableMultiTokenDecode: false,
              speculationMode: null,
              tokensPerReadback: 16,
              runtimeMirror: {
                batching: {
                  batchSize: 4,
                  readbackInterval: 4,
                  stopCheckMode: 'batch',
                  readbackMode: null,
                },
                decodeLoop: {
                  batchSize: 4,
                  readbackInterval: 4,
                  stopCheckMode: 'batch',
                  readbackMode: null,
                },
              },
              executionPlan: {
                id: 'resolved-parity-plan',
                batchSize: 4,
                readbackInterval: 4,
                stopCheckMode: 'batch',
                readbackMode: 'sequential',
                disableCommandBatching: false,
                ringTokens: null,
                ringStop: null,
                ringStaging: null,
              },
            },
          },
        },
      },
      transformersjs: { runs: [] },
    });
    assert.equal(measuredCadenceSection.dopplerDecodeCadence.readbackMode, 'sequential');
    assert.equal(measuredCadenceSection.dopplerDecodeCadence.runtimeMirror.decodeLoop.readbackMode, null);
    assert.equal(measuredCadenceSection.dopplerDecodeCadence.executionPlan.id, 'resolved-parity-plan');
    const bottleneckSection = buildCompareSection({
      cacheMode: 'warm',
      loadMode: 'opfs',
      maxTokens: 64,
      prefillTokenTarget: 64,
      promptContract: {
        promptRendered: 'unit prompt',
        enginesReceiveRenderedPrompt: true,
      },
      doppler: {
        result: {
          timing: {
            decodeMs: 100,
            decodeTokensPerSec: 120,
          },
          metrics: {
            avgPrefillTokens: 64,
            avgDecodeTokens: 64,
            generatedText: 'ok',
            batching: {
              requestedBatchTokens: { mean: 32, samples: 5 },
              effectiveBatchTokens: { mean: 32, samples: 5 },
              executedBatchTokens: { mean: 80, samples: 5 },
              resolvedBatchTokens: { mean: 64, samples: 5 },
              gpuSubmissions: { mean: 3, samples: 5 },
            },
            gpu: {
              decodeRecordMs: { mean: 35, samples: 5 },
              decodeRecordOps: { mean: 140, samples: 5 },
              decodeRecordPasses: { mean: 7, samples: 5 },
              decodeRecordUniqueOpLabels: 2,
              decodeRecordTopOps: [
                { label: 'matmul', count: 100, shareOfOps: 0.7142857142857143 },
                { label: 'sample', count: 40, shareOfOps: 0.2857142857142857 },
              ],
              decodeRecordMsPerOp: { mean: 0.25, samples: 5 },
              decodeRecordMsPerPass: { mean: 5, samples: 5 },
              decodeRecordPassesPerOp: { mean: 0.05, samples: 5 },
              decodeRecordMsPerExecutedBatchToken: { mean: 0.4375, samples: 5 },
              decodeRecordOpsPerExecutedBatchToken: { mean: 1.75, samples: 5 },
              decodeRecordPassesPerExecutedBatchToken: { mean: 0.0875, samples: 5 },
              decodeSubmitWaitMs: { mean: 50, samples: 5 },
              decodeReadbackWaitMs: { mean: 55, samples: 5 },
              decodeReadbackMapWaitMs: { mean: 42, samples: 5 },
              decodeReadbackCleanupMs: { mean: 2, samples: 5 },
              decodeReadbackCopyMs: { mean: 1, samples: 5 },
              decodeOrchestrationMs: { mean: 5, samples: 5 },
              decodeMs: { mean: 10, samples: 5 },
            },
          },
          memoryStats: {
            used: 9000,
            pool: {
              peakBytesAllocated: 12000,
              peakBytesRequested: 10000,
              currentBytesAllocated: 8000,
              currentBytesRequested: 7000,
              activeBuffers: 12,
              pooledBuffers: 3,
            },
            kvCache: {
              theoretical: 4096,
              allocated: 4096,
              used: 1024,
              efficiency: 0.25,
              seqLen: 64,
              maxSeqLen: 256,
              layout: 'contiguous',
              kvDtype: 'f16',
            },
          },
        },
      },
      transformersjs: {
        generatedText: 'ok',
        memoryInfo: {
          before: {
            usedJSHeapSize: 5000,
            totalJSHeapSize: 8000,
            jsHeapSizeLimit: 64000,
          },
          after: {
            usedJSHeapSize: 5600,
            totalJSHeapSize: 9000,
            jsHeapSizeLimit: 64000,
          },
        },
        runs: [
          {
            prefillTokens: 64,
            decodeTokens: 64,
          },
        ],
      },
    });
    assert.equal(bottleneckSection.pairedComparable, true);
    assert.deepEqual(bottleneckSection.outputParity, {
      schemaVersion: 1,
      status: 'match',
      reason: null,
      exactMatch: true,
      normalizedMatch: true,
      charMismatchIndex: -1,
      tokenMatch: {
        leftTokenCount: 1,
        rightTokenCount: 1,
        matchingPrefixTokens: 1,
        firstMismatchTokenIndex: -1,
      },
      tokenIdMatch: null,
      lengths: {
        dopplerChars: 2,
        transformersjsChars: 2,
      },
    });
    assert.deepEqual(bottleneckSection.dopplerBottleneck.componentsMs, {
      commandRecordMs: 35,
      submitWaitMs: 50,
      readbackWaitMs: 55,
      effectiveSubmitReadbackWaitMs: 55,
      readbackMapWaitMs: 42,
      readbackCleanupMs: 2,
      readbackCopyMs: 1,
      readbackUnattributedMs: 10,
      gpuTimestampMs: 10,
      submitReadbackSlackMs: 45,
      orchestrationMs: 5,
      residualMs: 5,
    });
    assert.equal(bottleneckSection.dopplerBottleneck.dominant.id, 'readback_map_wait');
    assert.equal(bottleneckSection.dopplerBottleneck.bottleneckClass, 'submit-readback-wait');
    assert.equal(bottleneckSection.dopplerBottleneck.dominant.shareOfDecode, 0.42);
    assert.equal(bottleneckSection.dopplerBottleneck.shares.commandRecord, 0.35);
    assert.equal(bottleneckSection.dopplerBottleneck.shares.effectiveSubmitReadbackWait, 0.55);
    assert.equal(bottleneckSection.dopplerBottleneck.shares.readbackMapWait, 0.42);
    assert.deepEqual(bottleneckSection.dopplerBottleneck.recording, {
      opCount: 140,
      passCount: 7,
      uniqueOpLabels: 2,
      msPerOp: 0.25,
      msPerPass: 5,
      passesPerOp: 0.05,
      msPerExecutedBatchToken: 0.438,
      opsPerExecutedBatchToken: 1.75,
      passesPerExecutedBatchToken: 0.088,
      topOps: [
        { label: 'matmul', count: 100, shareOfOps: 0.714 },
        { label: 'sample', count: 40, shareOfOps: 0.286 },
      ],
      semantics: {
        opCount: 'Logical compute dispatches recorded into batch-decode command buffers.',
        passCount: 'Actual WebGPU compute passes opened while recording batch-decode command buffers.',
        uniqueOpLabels: 'Number of distinct exact compute-pass labels recorded in batch-decode command buffers.',
        msPerOp: 'decodeRecordMs divided by logical recorded compute dispatches.',
        msPerPass: 'decodeRecordMs divided by actual WebGPU compute passes.',
        passesPerOp: 'Actual WebGPU compute passes divided by logical recorded compute dispatches.',
        msPerExecutedBatchToken: 'decodeRecordMs divided by batch-path tokens submitted for execution.',
        opsPerExecutedBatchToken: 'Logical recorded compute dispatches divided by batch-path tokens submitted for execution.',
        passesPerExecutedBatchToken: 'Actual WebGPU compute passes divided by batch-path tokens submitted for execution.',
        topOps: 'Highest-count exact compute-pass labels observed during command recording.',
      },
    });
    assert.deepEqual(bottleneckSection.dopplerBatchAccounting, {
      schemaVersion: 1,
      requestedBatchTokens: 32,
      effectiveBatchTokens: 32,
      executedBatchTokens: 80,
      resolvedBatchTokens: 64,
      outputDecodeTokens: 64,
      gpuSubmissions: 3,
      batchResolutionEfficiency: 0.8,
      outputEfficiency: 0.8,
      batchOverrunTokens: 16,
      outputOverrunTokens: 16,
      semantics: {
        executedBatchTokens: 'Total GPU batch-path decode tokens recorded/submitted for execution.',
        resolvedBatchTokens: 'Batch-path tokens retained by stop resolution before returning to the generation loop.',
        outputDecodeTokens: 'Decode tokens credited to output throughput metrics.',
      },
    });
    assert.deepEqual(bottleneckSection.memoryAccounting, {
      schemaVersion: 1,
      gpuMemoryComparable: false,
      doppler: {
        schemaVersion: 1,
        source: 'doppler-memoryStats',
        gpuMemoryComparable: true,
        usedBytes: 9000,
        pool: {
          peakBytesAllocated: 12000,
          peakBytesRequested: 10000,
          currentBytesAllocated: 8000,
          currentBytesRequested: 7000,
          activeBuffers: 12,
          pooledBuffers: 3,
        },
        kvCache: {
          allocatedBytes: 4096,
          usedBytes: 1024,
          theoreticalBytes: 4096,
          efficiency: 0.25,
          seqLen: 64,
          maxSeqLen: 256,
          layout: 'contiguous',
          kvDtype: 'f16',
        },
      },
      transformersjs: {
        schemaVersion: 1,
        source: 'browser-performance-memory',
        gpuMemoryComparable: false,
        usedJSHeapBeforeBytes: 5000,
        usedJSHeapAfterBytes: 5600,
        usedJSHeapDeltaBytes: 600,
        totalJSHeapBeforeBytes: 8000,
        totalJSHeapAfterBytes: 9000,
        jsHeapSizeLimitBytes: 64000,
      },
      comparability: {
        gpuMemory: false,
        reason: 'Doppler reports GPU buffer-pool/KV memory; Transformers.js exposes browser JS heap here, not WebGPU allocation residency.',
      },
    });
    const gateParitySection = buildCompareSection({
      cacheMode: 'warm',
      loadMode: 'opfs',
      maxTokens: 64,
      prefillTokenTarget: 64,
      promptContract: {
        promptRendered: 'unit prompt',
        enginesReceiveRenderedPrompt: true,
      },
      doppler: {
        result: {
          timing: {
            decodeTokensPerSec: 100,
          },
          metrics: {
            avgPrefillTokens: 64,
            avgDecodeTokens: 64,
            generatedText: 'ok',
          },
        },
      },
      transformersjs: {
        generatedText: 'ok',
        runs: [
          {
            prefillTokens: 64,
            decodeTokens: 64,
          },
        ],
      },
    });
    const promotableThroughputSection = buildCompareSection({
      cacheMode: 'warm',
      loadMode: 'opfs',
      maxTokens: 64,
      prefillTokenTarget: 64,
      promptContract: {
        promptRendered: 'unit prompt',
        enginesReceiveRenderedPrompt: true,
      },
      doppler: {
        result: {
          timing: {
            decodeTokensPerSec: 110,
          },
          metrics: {
            avgPrefillTokens: 64,
            avgDecodeTokens: 64,
            generatedText: 'ok',
            batching: {
              executedBatchTokens: { mean: 64, samples: 5 },
              resolvedBatchTokens: { mean: 64, samples: 5 },
              gpuSubmissions: { mean: 2, samples: 5 },
            },
          },
        },
      },
      transformersjs: {
        generatedText: 'ok',
        runs: [
          {
            prefillTokens: 64,
            decodeTokens: 64,
          },
        ],
      },
    });
    const promotableGate = resolveDopplerThroughputCadenceGate({
      paritySection: gateParitySection,
      throughputSection: promotableThroughputSection,
    });
    assert.equal(promotableGate.ok, true);
    assert.deepEqual(promotableGate.invalidReasons, []);
    assert.equal(promotableGate.observed.decodeTokensPerSecRatioVsParity, 1.1);
    assert.equal(promotableGate.observed.batchResolutionEfficiency, 1);

    const overrunGate = resolveDopplerThroughputCadenceGate({
      paritySection: gateParitySection,
      throughputSection: bottleneckSection,
    });
    assert.equal(overrunGate.ok, false);
    assert.ok(overrunGate.invalidReasons.includes('throughput-batch-resolution-efficiency-below-threshold'));
    assert.ok(overrunGate.invalidReasons.includes('throughput-batch-overrun-exceeds-threshold'));
    assert.equal(
      overrunGate.thresholds.minBatchResolutionEfficiency,
      benchmarkPolicy.promotionGates.throughputCadence.minBatchResolutionEfficiency
    );
    const invalidParityGate = resolveDopplerThroughputCadenceGate({
      paritySection: {
        ...gateParitySection,
        pairedComparable: false,
        invalidReason: 'output-parity-mismatch',
      },
      throughputSection: promotableThroughputSection,
    });
    assert.equal(invalidParityGate.ok, false);
    assert.ok(invalidParityGate.invalidReasons.includes('parity-section-not-comparable'));
    assert.equal(invalidParityGate.observed.parityInvalidReason, 'output-parity-mismatch');

    const fairnessClaimSection = buildCompareSection({
      cacheMode: 'warm',
      loadMode: 'opfs',
      maxTokens: 64,
      prefillTokenTarget: 64,
      promptContract: {
        promptRendered: 'unit prompt',
        enginesReceiveRenderedPrompt: true,
      },
      doppler: {
        result: {
          timing: {
            decodeTokensPerSec: 100,
          },
          metrics: {
            avgPrefillTokens: 64,
            avgDecodeTokens: 64,
            generatedText: 'same output',
          },
        },
      },
      transformersjs: {
        generatedText: 'same output',
        runs: [
          {
            prefillTokens: 64,
            decodeTokens: 64,
          },
        ],
      },
    });
    const browserFairness = buildCompareFairnessAudit({
      sections: {
        compute: {
          parity: fairnessClaimSection,
        },
      },
      compareLane: {
        declared: 'performance_comparable',
        allowNonComparableLane: false,
      },
      dopplerExecution: {
        requestedSurface: 'browser',
        commandSurface: 'browser',
      },
      dopplerFormat: 'rdrr',
      tjsFormat: 'onnx',
      tjsModelOverridden: false,
      dopplerModelSource: {
        source: 'quickstart-registry',
      },
    });
    assert.equal(browserFairness.claimGrade, true);
    assert.equal(browserFairness.releaseClaimable, true);
    assert.equal(browserFairness.localComparable, false);
    assert.equal(browserFairness.correctnessOk, true);
    assert.equal(browserFairness.invalidReason, null);
    assert.equal(browserFairness.primarySection, 'compute/parity');
    assert.equal(browserFairness.formatFairness.class, 'optimized-rdrr-vs-onnx');
    assert.equal(browserFairness.formatFairness.disclosureRequired, true);
    assert.equal(browserFairness.formatFairness.blocksClaim, false);

    const localFairness = buildCompareFairnessAudit({
      sections: {
        compute: {
          parity: fairnessClaimSection,
        },
      },
      compareLane: {
        declared: 'performance_comparable',
        allowNonComparableLane: false,
      },
      dopplerExecution: {
        requestedSurface: 'browser',
        commandSurface: 'browser',
      },
      dopplerFormat: 'safetensors',
      tjsFormat: 'safetensors',
      tjsModelOverridden: false,
      dopplerModelSource: {
        source: 'local',
      },
    });
    assert.equal(localFairness.claimGrade, true);
    assert.equal(localFairness.releaseClaimable, false);
    assert.equal(localFairness.localComparable, true);
    assert.equal(localFairness.formatFairness.class, 'neutral-safetensors-vs-safetensors');
    assert.equal(localFairness.formatFairness.disclosureRequired, false);

    const crossSurfaceFairness = buildCompareFairnessAudit({
      sections: {
        compute: {
          parity: fairnessClaimSection,
        },
      },
      compareLane: {
        declared: 'performance_comparable',
        allowNonComparableLane: false,
      },
      dopplerExecution: {
        requestedSurface: 'node',
        commandSurface: 'node',
      },
      dopplerFormat: 'rdrr',
      tjsFormat: 'onnx',
      tjsModelOverridden: false,
      dopplerModelSource: {
        source: 'quickstart-registry',
      },
    });
    assert.equal(crossSurfaceFairness.claimGrade, false);
    assert.equal(crossSurfaceFairness.releaseClaimable, false);
    assert.equal(crossSurfaceFairness.surfaceFairness.blocksClaim, true);
    assert.ok(crossSurfaceFairness.invalidReasons.includes('cross-surface-diagnostic:doppler-node-vs-transformersjs-browser'));

    const overriddenTjsFairness = buildCompareFairnessAudit({
      sections: {
        compute: {
          parity: fairnessClaimSection,
        },
      },
      compareLane: {
        declared: 'performance_comparable',
        allowNonComparableLane: false,
      },
      dopplerExecution: {
        requestedSurface: 'browser',
        commandSurface: 'browser',
      },
      dopplerFormat: 'rdrr',
      tjsFormat: 'onnx',
      tjsModelOverridden: true,
      dopplerModelSource: {
        source: 'quickstart-registry',
      },
    });
    assert.equal(overriddenTjsFairness.claimGrade, false);
    assert.ok(overriddenTjsFairness.invalidReasons.includes('transformersjs-model-overridden'));

    const throughputFairness = buildCompareFairnessAudit({
      sections: {
        compute: {
          parity: fairnessClaimSection,
          throughput: fairnessClaimSection,
          throughputCadenceGate: {
            ok: false,
            invalidReasons: ['missing-throughput-batch-accounting'],
          },
        },
      },
      compareLane: {
        declared: 'performance_comparable',
        allowNonComparableLane: false,
      },
      dopplerExecution: {
        requestedSurface: 'browser',
        commandSurface: 'browser',
      },
      dopplerFormat: 'rdrr',
      tjsFormat: 'onnx',
      tjsModelOverridden: false,
      dopplerModelSource: {
        source: 'quickstart-registry',
      },
    });
    assert.equal(throughputFairness.claimGrade, true, 'primary parity section remains claim-grade');
    assert.equal(throughputFairness.sections['compute/throughput'].claimGrade, false);
    assert.ok(
      throughputFairness.sections['compute/throughput'].invalidReasons
        .includes('throughput-cadence:missing-throughput-batch-accounting')
    );

    const zeroSampleTimestampSection = buildCompareSection({
      cacheMode: 'warm',
      loadMode: 'opfs',
      maxTokens: 64,
      prefillTokenTarget: 64,
      promptContract: {
        promptRendered: 'unit prompt',
        enginesReceiveRenderedPrompt: true,
      },
      doppler: {
        result: {
          timing: {
            decodeMs: 100,
          },
          metrics: {
            avgPrefillTokens: 64,
            avgDecodeTokens: 64,
            generatedText: 'ok',
            gpu: {
              decodeSubmitWaitMs: { mean: 60, samples: 5 },
              decodeReadbackWaitMs: { mean: 50, samples: 5 },
              decodeMs: { mean: 0, samples: 0 },
            },
          },
        },
      },
      transformersjs: {
        generatedText: 'ok',
        runs: [
          {
            prefillTokens: 64,
            decodeTokens: 64,
          },
        ],
      },
    });
    assert.equal(zeroSampleTimestampSection.dopplerBottleneck.componentsMs.gpuTimestampMs, null);
    assert.equal(zeroSampleTimestampSection.dopplerBottleneck.componentsMs.submitReadbackSlackMs, null);
    assert.equal(zeroSampleTimestampSection.dopplerBottleneck.componentsMs.readbackMapWaitMs, null);
    assert.equal(zeroSampleTimestampSection.dopplerBottleneck.dominant.id, 'submit_readback_wait');

    const diagnosticSharedContract = buildDiagnosticSharedBenchmarkContract(buildSharedBenchmarkContract({
      prompt: 'unit prompt',
      maxTokens: 64,
      warmupRuns: 4,
      timedRuns: 9,
      seed: 0,
      sampling: {
        temperature: 0,
        topK: 1,
        topP: 1,
        repetitionPenalty: 1,
        greedyThreshold: 0.01,
      },
      useChatTemplate: false,
      loadMode: 'opfs',
      promptContract: {
        promptRaw: 'unit prompt',
        promptRendered: 'unit prompt',
        promptRenderedSha256: 'unit-sha',
        enginesReceiveRenderedPrompt: true,
      },
    }));
    assert.equal(diagnosticSharedContract.warmupRuns, 0);
    assert.equal(diagnosticSharedContract.timedRuns, 1);
    const profileSummary = summarizeDopplerDecodeProfileSteps([
      {
        timings: {
          attention: 10,
          'matmul:lm_head': 5,
        },
      },
      {
        timings: {
          attention: 11,
          rmsnorm: 2,
        },
      },
    ]);
    assert.equal(profileSummary.profileStepCount, 2);
    assert.equal(profileSummary.profileOperationCount, 4);
    assert.equal(profileSummary.averageOperationsPerStep, 2);
    assert.equal(profileSummary.totalTimestampMs, 28);
    assert.deepEqual(profileSummary.topOperations[0], {
      operation: 'attention',
      operationKind: 'attention',
      totalMs: 21,
      count: 2,
    });
    const diagnostic = buildDopplerBottleneckDiagnostic({
      sourceSection: 'compute/throughput',
      sharedContract: diagnosticSharedContract,
      doppler: {
        request: matchingDopplerResult.request,
        result: {
          timing: {
            decodeMs: 100,
            decodeTokensPerSec: 120,
          },
          metrics: {
            firstTokenMs: 20,
            modelLoadMs: 50,
            batching: {
              executedBatchTokens: { mean: 64, samples: 5 },
              resolvedBatchTokens: { mean: 64, samples: 5 },
            },
            gpu: {
              decodeRecordMs: { mean: 35, samples: 5 },
              decodeRecordOps: { mean: 140, samples: 5 },
              decodeRecordPasses: { mean: 7, samples: 5 },
              decodeRecordUniqueOpLabels: 2,
              decodeRecordTopOps: [
                { label: 'matmul', count: 100, shareOfOps: 0.7142857142857143 },
                { label: 'sample', count: 40, shareOfOps: 0.2857142857142857 },
              ],
              decodeRecordMsPerOp: { mean: 0.25, samples: 5 },
              decodeRecordMsPerPass: { mean: 5, samples: 5 },
              decodeRecordPassesPerOp: { mean: 0.05, samples: 5 },
              decodeRecordMsPerExecutedBatchToken: { mean: 0.4375, samples: 5 },
              decodeRecordOpsPerExecutedBatchToken: { mean: 1.75, samples: 5 },
              decodeRecordPassesPerExecutedBatchToken: { mean: 0.109375, samples: 5 },
              decodeSubmitWaitMs: { mean: 50, samples: 5 },
              decodeReadbackWaitMs: { mean: 55, samples: 5 },
              decodeReadbackMapWaitMs: { mean: 42, samples: 5 },
              decodeReadbackCleanupMs: { mean: 2, samples: 5 },
              decodeReadbackCopyMs: { mean: 1, samples: 5 },
              decodeOrchestrationMs: { mean: 5, samples: 5 },
              decodeMs: { mean: 28, samples: 5 },
            },
            decodeProfileSteps: [
              {
                timings: {
                  attention: 10,
                  'matmul:lm_head': 5,
                },
              },
              {
                timings: {
                  attention: 11,
                  rmsnorm: 2,
                },
              },
            ],
          },
        },
      },
    });
    assert.equal(diagnostic.kind, 'doppler-bottleneck-profile');
    assert.equal(diagnostic.claimUse, 'diagnostic-only');
    assert.equal(diagnostic.sourceSection, 'compute/throughput');
    assert.equal(diagnostic.complete, true);
    assert.equal(diagnostic.invalidReason, null);
    assert.deepEqual(diagnostic.invalidReasons, []);
    assert.equal(diagnostic.command.dopplerCommand, 'debug');
    assert.equal(diagnostic.command.workload, 'inference');
    assert.equal(diagnostic.command.profileEnabled, true);
    assert.equal(diagnostic.command.warmupRuns, 0);
    assert.equal(diagnostic.command.timedRuns, 1);
    assert.equal(diagnostic.decodeTokensPerSec, 120);
    assert.equal(diagnostic.firstTokenMs, 20);
    assert.equal(diagnostic.modelLoadMs, 50);
    assert.equal(diagnostic.dopplerBottleneck.componentsMs.gpuTimestampMs, 28);
    assert.equal(diagnostic.dopplerBottleneck.componentsMs.submitReadbackSlackMs, 27);
    assert.equal(diagnostic.dopplerBottleneck.recording.msPerOp, 0.25);
    assert.equal(diagnostic.dopplerBottleneck.recording.passCount, 7);
    assert.equal(diagnostic.dopplerBottleneck.recording.topOps[0].label, 'matmul');
    assert.equal(diagnostic.profile.topOperations[0].operation, 'attention');
    assert.equal(diagnostic.coverage.completeDecodeTarget, true);
    assert.equal(diagnostic.coverage.expectedBatchDecodeTokens, 63);
    assert.equal(diagnostic.coverage.warning, null);
    const partialDiagnostic = buildDopplerBottleneckDiagnostic({
      sourceSection: 'compute/throughput',
      sharedContract: {
        ...diagnosticSharedContract,
        maxTokens: 128,
      },
      doppler: {
        request: matchingDopplerResult.request,
        result: {
          timing: {
            decodeMs: 100,
            decodeTokensPerSec: 60,
          },
          metrics: {
            batching: {
              executedBatchTokens: { mean: 31, samples: 1 },
              resolvedBatchTokens: { mean: 31, samples: 1 },
            },
            gpu: {
              decodeMs: { mean: 10, samples: 1 },
            },
          },
        },
      },
    });
    assert.equal(partialDiagnostic.coverage.completeDecodeTarget, false);
    assert.equal(partialDiagnostic.coverage.expectedBatchDecodeTokens, 127);
    assert.equal(partialDiagnostic.coverage.resolvedBatchTokens, 31);
    assert.equal(partialDiagnostic.coverage.profileCoverage, 0.244);
    assert.equal(partialDiagnostic.coverage.warning, 'diagnostic-run-ended-before-source-decode-target');
    assert.equal(partialDiagnostic.complete, false);
    assert.equal(partialDiagnostic.invalidReason, 'diagnostic-run-ended-before-source-decode-target');
    assert.deepEqual(partialDiagnostic.invalidReasons, [
      'diagnostic-run-ended-before-source-decode-target',
      'missing-doppler-decode-profile-steps',
    ]);
    const diagnosticOverlay = buildDopplerBottleneckDiagnosticRuntimeOverlay();
    assert.equal(diagnosticOverlay.shared.debug.profiler.enabled, true);
    assert.equal(diagnosticOverlay.shared.debug.profiler.logEveryDecodeSteps, 1);
    assert.equal(diagnosticOverlay.shared.tooling?.diagnostics, undefined);
    assert.throws(
      () => assertDopplerDecodeCadence(
        {
          request: {
            runtimeConfig: {
              inference: {
                generation: {
                  disableMultiTokenDecode: false,
                },
                batching: {
                  batchSize: 4,
                  readbackInterval: 6,
                  stopCheckMode: 'batch',
                },
                session: {
                  decodeLoop: {
                    batchSize: 8,
                    readbackInterval: 6,
                    stopCheckMode: 'batch',
                    disableCommandBatching: false,
                  },
                },
              },
            },
          },
        },
        customCadence,
        'unit/custom'
      ),
      /inference\.batching\.batchSize/
    );
  }
  assert.ok(gemma3Profile, 'compare config must include gemma-3-270m-it-q4k-ehf16-af32');
  assert.equal(
    gemma3Profile?.dopplerRuntimeProfileByDecodeProfile?.throughput,
    'profiles/gemma3-270m-q4k-throughput-overlapped-probe'
  );
  assert.equal(
    gemma3Profile?.dopplerRuntimeProfileByDecodeProfile?.custom,
    'profiles/gemma3-270m-q4k-throughput-overlapped-probe'
  );
  assert.equal(gemma3Q4kThroughputProfile.extends, 'profiles/throughput');
  assert.equal(gemma3Q4kThroughputProfile.intent, 'calibrate');
  assert.equal(gemma3Q4kThroughputProfile.stability, 'experimental');
  const throughputDecodeProfile = benchmarkPolicy.decodeProfiles.profiles.throughput;
  const expectedGemma3Q4kThroughputRuntime = {
    inference: {
      generation: {
        disableMultiTokenDecode: throughputDecodeProfile.disableMultiTokenDecode,
      },
      batching: {
        batchSize: throughputDecodeProfile.batchSize,
        readbackInterval: throughputDecodeProfile.readbackInterval,
        stopCheckMode: throughputDecodeProfile.stopCheckMode,
        readbackMode: 'overlapped',
      },
      session: {
        prefillChunkSubmitMode: 'sync',
        retainQ4KMaterialization: true,
        useWideTileQ4KPrefill: true,
        useSandwichRMSNormPairFusion: true,
        usePostFfnNextInputRMSNormPairFusion: true,
        useFusedQKVSplitQKNormRoPE: true,
        decodeLoop: {
          batchSize: throughputDecodeProfile.batchSize,
          stopCheckMode: throughputDecodeProfile.stopCheckMode,
          readbackInterval: throughputDecodeProfile.readbackInterval,
          readbackMode: 'overlapped',
          ringTokens: 2,
          ringStop: 1,
          ringStaging: 2,
          disableCommandBatching: false,
        },
      },
    },
  };
  assert.deepEqual(gemma3Q4kThroughputProfile.runtime, expectedGemma3Q4kThroughputRuntime);
  assert.ok(qwen08Profile, 'compare config must include qwen-3-5-0-8b-q4k-ehaf16');
  assert.equal(qwen08Profile.defaultDopplerSurface, 'browser');
  assert.equal(
    qwen08Profile?.dopplerRuntimeProfileByDecodeProfile?.throughput,
    'profiles/throughput'
  );
  assert.equal(qwen08Profile.compareLane, 'performance_comparable');
  assert.equal(qwen08Profile.compareLaneReason, null);

  assert.ok(qwen2Profile, 'compare config must include qwen-3-5-2b-q4k-ehaf16');
  assert.equal(qwen2Profile.defaultDopplerSurface, 'browser');
  assert.equal(
    qwen2Profile?.dopplerRuntimeProfileByDecodeProfile?.throughput,
    'profiles/throughput'
  );
  assert.equal(qwen2Profile.compareLane, 'performance_comparable');
  assert.equal(qwen2Profile.compareLaneReason, null);
  assert.equal(qwen2Profile.defaultLoadMode, 'http');
  assert.match(qwen2Profile.defaultLoadModeReason, /strict offline/i);

  assert.ok(gemma4Profile, 'compare config must include gemma-4-e2b-it-q4k-ehf16-af32');
  assert.equal(gemma4Profile.defaultDopplerSurface, 'browser');
  assert.equal(gemma4Profile.defaultUseChatTemplate, true);
  assert.equal(
    gemma4Profile?.dopplerRuntimeProfileByDecodeProfile?.throughput,
    'profiles/throughput'
  );
  assert.equal(gemma4Profile.compareLane, 'performance_comparable');
  assert.match(gemma4Profile.compareLaneReason, /not exact-match/i);
  assert.ok(gemma4Int4PleProfile, 'compare config must include gemma-4-e2b-it-q4k-ehf16-af32-int4ple');
  assert.equal(
    gemma4Int4PleProfile?.dopplerRuntimeProfileByDecodeProfile?.throughput,
    'profiles/throughput'
  );
  assert.ok(gemma4Af16Int4PleProfile, 'compare config must include gemma-4-e2b-it-q4k-ehf16-af16-int4ple');
  assert.equal(gemma4Af16Int4PleProfile.defaultDopplerSource, 'local');
  assert.equal(gemma4Af16Int4PleProfile.defaultDopplerSurface, 'browser');
  assert.equal(gemma4Af16Int4PleProfile.defaultLoadMode, 'http');
  assert.match(gemma4Af16Int4PleProfile.defaultLoadModeReason, /weights-ref sibling/i);
  assert.equal(gemma4Af16Int4PleProfile.defaultUseChatTemplate, true);
  assert.equal(gemma4Af16Int4PleProfile.compareLane, 'performance_comparable');
  assert.match(gemma4Af16Int4PleProfile.compareLaneReason, /local compute-throughput evidence/i);
  assert.equal(
    gemma4Af16Int4PleProfile?.dopplerRuntimeProfileByDecodeProfile?.throughput,
    'profiles/throughput'
  );
  assert.ok(gemma4Catalog, 'models/catalog.json must include gemma-4-e2b-it-q4k-ehf16-af32');
  assert.equal(gemma4Catalog?.vendorBenchmark?.transformersjs?.repoId, gemma4Profile.defaultTjsModelId);
  assert.equal(gemma4Catalog?.vendorBenchmark?.transformersjs?.dtype, 'q4f16');

  for (const profile of compareConfig.modelProfiles) {
    assert.ok(['performance_comparable', 'capability_only'].includes(profile.compareLane));
    if (profile.compareLane === 'capability_only') {
      assert.equal(typeof profile.compareLaneReason, 'string');
      assert.ok(profile.compareLaneReason.length > 0);
    }

    if (profile.defaultDopplerSource === 'quickstart-registry') {
      assert.equal(profile.modelBaseDir, null);
      assert.ok(
        quickstartIds.has(profile.dopplerModelId),
        `${profile.dopplerModelId}: quickstart compare profiles must exist in src/client/doppler-registry.json`
      );
      continue;
    }

    assert.equal(profile.defaultDopplerSource, 'local');
    if (profile?.modelBaseDir !== 'local') {
      continue;
    }
    const manifestPath = path.join(repoRoot, 'models', 'local', profile.dopplerModelId, 'manifest.json');
    let manifest = null;
    try {
      manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
    } catch (error) {
      if (error?.code !== 'ENOENT') {
        throw error;
      }
    }
    if (manifest) {
      assert.equal(manifest.modelId, profile.dopplerModelId);
      continue;
    }
    assert.equal(typeof profile.dopplerModelId, 'string');
    assert.ok(profile.dopplerModelId.trim().length > 0);
  }

  const knownBadByModel = benchmarkPolicy?.kernelPathPolicy?.knownBadByModel ?? {};
  for (const modelId of Object.keys(knownBadByModel)) {
    const localManifestPath = path.join(repoRoot, 'models', 'local', modelId, 'manifest.json');
    let localExists = true;
    try {
      await fs.access(localManifestPath);
    } catch {
      localExists = false;
    }
    assert.ok(
      localExists || catalogIds.has(modelId) || compareProfileIds.has(modelId),
      `benchmark-policy knownBadByModel.${modelId} must resolve to a local manifest, compare profile, or catalog model`
    );
  }

  const compareRuntimePolicy = mergeKernelPathPolicy(
    undefined,
    benchmarkPolicy?.kernelPathPolicy?.compareRuntime
  );
  assert.equal(compareRuntimePolicy.mode, 'capability-aware');
  assert.deepEqual(compareRuntimePolicy.sourceScope, ['model', 'manifest', 'config']);
  assert.deepEqual(compareRuntimePolicy.allowSources, ['model', 'manifest', 'config']);
  assert.equal(compareRuntimePolicy.onIncompatible, 'remap');
  assert.equal(benchmarkPolicy?.browser?.compareChannelByPlatform?.darwin, 'chromium');
  assert.equal(benchmarkPolicy?.browser?.compareChannelByPlatform?.linux, 'chromium');
  assert.equal(benchmarkPolicy?.browser?.compareChannelByPlatform?.win32, 'chromium');
  assert.equal(benchmarkPolicy?.decodeProfiles?.profiles?.parity?.stopCheckMode, 'batch');
  assert.equal(benchmarkPolicy?.decodeProfiles?.profiles?.parity?.batchSize, 4);
  assert.equal(benchmarkPolicy?.decodeProfiles?.profiles?.parity?.readbackInterval, 4);
  assert.equal(benchmarkPolicy?.decodeProfiles?.profiles?.throughput?.stopCheckMode, 'batch');
  assert.equal(benchmarkPolicy?.decodeProfiles?.profiles?.throughput?.batchSize, 8);
  assert.equal(benchmarkPolicy?.decodeProfiles?.profiles?.throughput?.readbackInterval, 8);
  assert.deepEqual(
    benchmarkPolicy?.doppler?.loadModeRuntimeOverlays?.http?.runtimeConfig?.loading?.distribution?.sourceOrder,
    ['http']
  );
  assert.equal(
    benchmarkPolicy?.doppler?.loadModeRuntimeOverlays?.http?.runtimeConfig?.loading?.shardCache?.verifyHashes,
    false
  );
  assert.equal(
    benchmarkPolicy?.doppler?.loadModeRuntimeOverlays?.http?.runtimeConfig?.loading?.shardCache?.rangeCacheBlockBytes,
    67108864
  );
  assert.equal(
    benchmarkPolicy?.doppler?.loadModeRuntimeOverlays?.http?.runtimeConfig?.loading?.shardCache?.rangeCacheMaxBytes,
    536870912
  );
  assert.equal(
    benchmarkPolicy?.doppler?.loadModeRuntimeOverlays?.http?.runtimeConfig?.loading?.shardCache?.maxConcurrentLoads,
    8
  );
  assert.equal(
    benchmarkPolicy?.doppler?.loadModeRuntimeOverlays?.http?.runtimeConfig?.loading?.prefetch?.allowRangeLoaderPrefetch,
    true
  );
  assert.equal(
    benchmarkPolicy?.doppler?.loadModeRuntimeOverlays?.http?.runtimeConfig?.loading?.prefetch?.maxShards,
    8
  );
}

{
  const catalogBundle = await loadModelCatalogBundle();
  const gemma4Benchmark = resolveCatalogTransformersjsBenchmarkTarget(
    catalogBundle,
    'gemma-4-e2b-it-q4k-ehf16-af32',
    null
  );
  assert.equal(gemma4Benchmark?.repoId, 'onnx-community/gemma-4-E2B-it-ONNX');
  assert.equal(gemma4Benchmark?.dtype, 'q4f16');
  assert.equal(gemma4Benchmark?.source, 'catalog-model');

  const repoOverrideBenchmark = resolveCatalogTransformersjsBenchmarkTarget(
    catalogBundle,
    'unit-missing-model',
    'onnx-community/gemma-4-E2B-it-ONNX'
  );
  assert.equal(repoOverrideBenchmark?.repoId, 'onnx-community/gemma-4-E2B-it-ONNX');
  assert.equal(repoOverrideBenchmark?.dtype, 'q4f16');
  assert.equal(repoOverrideBenchmark?.source, 'catalog-repo');
}

{
  const result = runCompareEngines(['--help']);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout || renderCompareUsage(), /--doppler-surface <surface>\s+auto\|node\|browser\|bun/);
  assert.match(result.stdout || renderCompareUsage(), /--doppler-stop-check-mode <batch\|per-token>/);
  assert.match(result.stdout || renderCompareUsage(), /--doppler-bottleneck-profile <on\|off>/);
}

{
  assert.deepEqual(resolveDopplerExecutionIdentity('node', 'rdrr'), {
    requestedSurface: 'node',
    commandSurface: 'node',
    cliExecutor: 'node',
    format: 'rdrr',
    commandSurfaceReason: 'matches-requested-surface',
  });
  assert.deepEqual(resolveDopplerExecutionIdentity('bun', 'rdrr'), {
    requestedSurface: 'bun',
    commandSurface: 'node',
    cliExecutor: 'bun',
    format: 'rdrr',
    commandSurfaceReason: 'bun-executor-node-command-surface',
  });
  assert.deepEqual(resolveDopplerExecutionIdentity('browser', 'safetensors'), {
    requestedSurface: 'browser',
    commandSurface: 'node',
    cliExecutor: 'node',
    format: 'safetensors',
    commandSurfaceReason: 'safetensors-format-node-command-surface',
  });
}

{
  const defaults = normalizeCompareLoadModeDefaults({
    warmLoadMode: 'opfs',
    coldLoadMode: 'http',
  });
  assert.deepEqual(defaults, {
    warm: 'opfs',
    cold: 'http',
  });
  assert.deepEqual(resolveCompareLoadModes(null, defaults), {
    warm: 'opfs',
    cold: 'http',
  });
  assert.deepEqual(resolveCompareLoadModes('memory', defaults), {
    warm: 'memory',
    cold: 'memory',
  });
}

{
  const flags = parseCompareArgs(['--use-chat-template', 'off']);
  assert.equal(flags['use-chat-template'], 'off');
  const promptContract = renderComparePrompt({
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    prompt: 'The sky is clear.',
    useChatTemplate: true,
  });
  assert.equal(promptContract.promptRaw, 'The sky is clear.');
  assert.equal(promptContract.promptRenderer, 'gemma4-compare-template');
  assert.equal(promptContract.chatTemplateSource, 'compare-engines');
  assert.match(promptContract.promptRendered, /^<bos><\|turn\>user\nThe sky is clear\.<turn\|>\n<\|turn\>model\n$/);
  assert.equal(promptContract.enginesReceiveRenderedPrompt, true);
  const sharedContract = buildSharedBenchmarkContract({
    prompt: promptContract.promptRendered,
    maxTokens: 4,
    warmupRuns: 0,
    timedRuns: 1,
    seed: 7,
    sampling: {
      temperature: 0,
      topK: 1,
      topP: 1,
    },
    useChatTemplate: parseCompareOnOff(flags['use-chat-template'], false, '--use-chat-template'),
    promptContract,
  });
  assert.equal(sharedContract.useChatTemplate, false);
  assert.equal(sharedContract.promptRaw, 'The sky is clear.');
  assert.equal(sharedContract.promptContract.promptRendered, promptContract.promptRendered);
  assert.equal(sharedContract.sampling.repetitionPenalty, 1);
  assert.equal(sharedContract.sampling.greedyThreshold, 0.01);
}

{
  const renderer = resolveCompareOwnedPromptRenderer({
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    useChatTemplate: true,
  });
  assert.equal(typeof renderer, 'function');
  assert.match(renderer('The sky is clear.'), /^<bos><\|turn\>user\nThe sky is clear\.<turn\|>\n<\|turn\>model\n$/);

  const qwenRenderer = resolveCompareOwnedPromptRenderer({
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    useChatTemplate: true,
  });
  assert.equal(qwenRenderer, null);
}

{
  const parsed = parseJsonBlock('[info] noisy line\n{\n  "ok": true,\n  "value": 3\n}\n', 'unit-json');
  assert.deepEqual(parsed, { ok: true, value: 3 });
}

{
  const secret = `hf_${'unitSecretToken123'}`;
  const redacted = redactSecrets(`runner.html?hfToken=${secret}:12 Authorization: Bearer ${secret}`);
  assert.doesNotMatch(redacted, new RegExp(secret));
  assert.match(redacted, /hfToken=<redacted>/);
  assert.match(redacted, /Authorization: Bearer <redacted>/i);

  assert.throws(
    () => parseJsonBlock(`bad output runner.html?hfToken=${secret}`, 'secret-tail'),
    (error) => {
      assert.doesNotMatch(String(error.message), new RegExp(secret));
      assert.match(String(error.message), /hfToken=<redacted>/);
      return true;
    }
  );
}

{
  const section = buildCompareSection({
    cacheMode: 'warm',
    loadMode: 'opfs',
    doppler: { result: { timing: { decodeTokensPerSec: 1 } } },
    transformersjs: {
      failed: true,
      error: {
        message: 'fetch failed',
      },
    },
  });
  assert.equal(section.cacheMode, 'warm');
  assert.equal(section.loadMode, 'opfs');
  assert.equal(section.pairedComparable, false);
  assert.equal(section.invalidReason, 'transformersjs-failed');
  assert.deepEqual(section.outputParity, {
    schemaVersion: 1,
    status: 'unavailable',
    reason: 'engine-run-missing-or-failed',
    exactMatch: null,
    normalizedMatch: null,
    charMismatchIndex: null,
    tokenMatch: null,
    tokenIdMatch: null,
    lengths: {
      dopplerChars: null,
      transformersjsChars: null,
    },
  });
}

{
  const promptContract = renderComparePrompt({
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    prompt: 'unit prompt',
    useChatTemplate: true,
  });
  const section = buildCompareSection({
    cacheMode: 'warm',
    loadMode: 'opfs',
    maxTokens: 64,
    promptContract,
    doppler: {
      result: {
        result: {
          metrics: {
            avgPrefillTokens: 64,
            avgDecodeTokens: 64,
            generatedText: 'That sounds lovely.',
          },
        },
      },
    },
    transformersjs: {
      generatedText: '',
      runs: [
        {
          prefillTokens: 64,
          decodeTokens: 0,
        },
      ],
    },
    prefillTokenTarget: 64,
  });
  assert.equal(section.pairedComparable, false);
  assert.equal(section.invalidReason, 'transformersjs-invalid-zero-decode-tokens');
  assert.equal(section.promptContract.promptTokenCount, 64);
  assert.equal(section.decodeValidity.code, 'INVALID_BENCHMARK_ZERO_DECODE_TOKENS');
}

{
  const section = buildCompareSection({
    cacheMode: 'warm',
    loadMode: 'opfs',
    maxTokens: 64,
    promptContract: {
      promptRendered: 'unit prompt',
      enginesReceiveRenderedPrompt: false,
    },
    doppler: {
      result: {
        result: {
          metrics: {
            avgPrefillTokens: 64,
            avgDecodeTokens: 64,
            generatedText: 'ok',
          },
        },
      },
    },
    transformersjs: {
      generatedText: 'ok',
      runs: [
        {
          prefillTokens: 64,
          decodeTokens: 64,
        },
      ],
    },
    prefillTokenTarget: 64,
  });
  assert.equal(section.pairedComparable, false);
  assert.equal(section.invalidReason, 'prompt-rendering-not-shared');
}

{
  const resolved = await buildTokenAccurateSyntheticPrompt({
    prefillTokens: 8,
    countPromptTokens: async (prompt) => {
      const words = String(prompt || '').trim();
      if (words.length === 0) {
        return 2;
      }
      return 2 + words.split(/\s+/).length;
    },
  });
  assert.equal(resolved.prefillTokens, 8);
  assert.match(resolved.prompt, /\w/);
}

{
  await assert.rejects(
    () => buildTokenAccurateSyntheticPrompt({
      prefillTokens: 5,
      countPromptTokens: async (prompt) => {
        const words = String(prompt || '').trim();
        if (words.length === 0) {
          return 2;
        }
        return 2 + (words.split(/\s+/).length * 2);
      },
    }),
    /Could not synthesize an exact 5-token prompt/i
  );
}

{
  const section = buildCompareSection({
    cacheMode: 'warm',
    loadMode: 'opfs',
    prefillTokenTarget: 64,
    doppler: {
      result: {
        result: {
          metrics: {
            avgPrefillTokens: 70,
          },
        },
      },
    },
    transformersjs: {
      runs: [
        {
          prefillTokens: 64,
        },
      ],
    },
  });
  assert.equal(section.pairedComparable, false);
  assert.equal(section.invalidReason, 'prompt-token-count-mismatch');
  assert.equal(section.promptTokens.target, 64);
  assert.equal(section.promptTokens.doppler, 70);
  assert.equal(section.promptTokens.transformersjs, 64);
  assert.equal(section.promptTokens.ok, false);
  assert.equal(section.promptTokens.invalidReason, 'prompt-token-count-mismatch');
  assert.equal(section.promptTokens.tokenizerDelta, 6);
  assert.equal(section.promptTokens.pairedComparable, false);
  assert.equal(section.promptTokens.toleratedTokenizerDelta, null);
}

// Tokenizer-delta tolerance: ±1 difference between Doppler and Transformers.js
// tokenizers on an identical compare-rendered prompt is tolerated when the
// rest of the compare contract is already clean (compare owns rendering,
// both sides decoded non-zero, both sides produced non-empty text).
{
  const promptContract = renderComparePrompt({
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    prompt: 'the sky is clear',
    useChatTemplate: true,
  });
  const section = buildCompareSection({
    cacheMode: 'warm',
    loadMode: 'opfs',
    maxTokens: 64,
    promptContract,
    prefillTokenTarget: 64,
    doppler: {
      result: {
        result: {
          metrics: {
            avgPrefillTokens: 63,
            avgDecodeTokens: 64,
            generatedText: 'Coherent Doppler output.',
          },
        },
      },
    },
    transformersjs: {
      generatedText: 'Coherent Doppler output.',
      runs: [
        {
          prefillTokens: 64,
          decodeTokens: 64,
        },
      ],
    },
  });
  assert.equal(section.pairedComparable, true, 'tokenizer delta of 1 must be tolerated when every other gate passes');
  assert.equal(section.invalidReason, null);
  assert.equal(section.outputParity.status, 'match');
  assert.equal(section.outputParity.exactMatch, true);
  assert.equal(section.outputParity.normalizedMatch, true);
  assert.deepEqual(section.outputParity.tokenMatch, {
    leftTokenCount: 3,
    rightTokenCount: 3,
    matchingPrefixTokens: 3,
    firstMismatchTokenIndex: -1,
  });
  assert.equal(section.outputParity.tokenIdMatch, null);
  assert.equal(section.promptTokens.ok, false, 'strict ok stays false so the evidence is preserved');
  assert.equal(section.promptTokens.invalidReason, 'prompt-token-count-mismatch');
  assert.equal(section.promptTokens.tokenizerDelta, 1);
  assert.equal(section.promptTokens.pairedComparable, true);
  assert.ok(section.promptTokens.toleratedTokenizerDelta, 'toleration record must be present');
  assert.equal(section.promptTokens.toleratedTokenizerDelta.delta, 1);
  assert.equal(section.promptTokens.toleratedTokenizerDelta.maxAllowed, 1);
  assert.equal(section.promptTokens.toleratedTokenizerDelta.doppler, 63);
  assert.equal(section.promptTokens.toleratedTokenizerDelta.transformersjs, 64);
}

{
  const promptContract = renderComparePrompt({
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    prompt: 'the sky is clear',
    useChatTemplate: true,
  });
  const section = buildCompareSection({
    cacheMode: 'warm',
    loadMode: 'opfs',
    maxTokens: 64,
    promptContract,
    prefillTokenTarget: 64,
    outputParityPolicy: {
      schemaVersion: 1,
      requireMatch: true,
      matchMode: 'exact-or-normalized',
      reason: 'deterministic-greedy-sampling',
    },
    doppler: {
      result: {
        result: {
          metrics: {
            avgPrefillTokens: 64,
            avgDecodeTokens: 64,
            generatedText: 'Coherent Doppler output.',
          },
        },
      },
    },
    transformersjs: {
      generatedText: 'Coherent TJS output.',
      runs: [
        {
          prefillTokens: 64,
          decodeTokens: 64,
        },
      ],
    },
  });
  assert.equal(section.pairedComparable, false);
  assert.equal(section.invalidReason, 'output-parity-mismatch');
  assert.equal(section.outputParity.status, 'mismatch');
  assert.equal(section.outputParityPolicy.requireMatch, true);
}

{
  const section = buildCompareSection({
    cacheMode: 'warm',
    loadMode: 'opfs',
    maxTokens: 64,
    outputParityPolicy: {
      schemaVersion: 1,
      requireMatch: true,
      matchMode: 'exact-or-normalized',
      reason: 'deterministic-greedy-sampling',
    },
    doppler: {
      result: {
        metrics: {
          avgPrefillTokens: 64,
          avgDecodeTokens: 64,
          generatedText: 'same text with hidden token drift',
          referenceTranscript: {
            tokens: {
              ids: [10, 20, 30, 40],
            },
          },
        },
      },
    },
    transformersjs: {
      generatedText: 'same text with hidden token drift',
      generatedTokenIds: [10, 20, 31, 40],
      runs: [
        {
          prefillTokens: 64,
          decodeTokens: 64,
        },
      ],
    },
  });
  assert.equal(section.outputParity.status, 'match');
  assert.deepEqual(section.outputParity.tokenIdMatch, {
    leftTokenCount: 4,
    rightTokenCount: 4,
    matchingPrefixTokens: 2,
    firstMismatchTokenIndex: 2,
    firstMismatch: {
      doppler: 30,
      transformersjs: 31,
    },
  });
}

// Tolerance gate must NOT fire when decodeValidity fails — zero decode tokens
// on one side means there's no evidence the prompt path actually executed end
// to end, so a ±1 tokenizer delta is no longer an innocuous divergence.
{
  const promptContract = renderComparePrompt({
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    prompt: 'the sky is clear',
    useChatTemplate: true,
  });
  const section = buildCompareSection({
    cacheMode: 'warm',
    loadMode: 'opfs',
    maxTokens: 64,
    promptContract,
    prefillTokenTarget: 64,
    doppler: {
      result: {
        result: {
          metrics: {
            avgPrefillTokens: 63,
            avgDecodeTokens: 64,
            generatedText: 'Coherent Doppler output.',
          },
        },
      },
    },
    transformersjs: {
      generatedText: '',
      runs: [
        {
          prefillTokens: 64,
          decodeTokens: 0,
        },
      ],
    },
  });
  assert.equal(section.pairedComparable, false);
  assert.equal(section.invalidReason, 'transformersjs-invalid-zero-decode-tokens');
  assert.equal(section.promptTokens.pairedComparable, false, 'gate must not relax when decodeValidity fails');
  assert.equal(section.promptTokens.toleratedTokenizerDelta, null);
}

// Tolerance gate must NOT fire when enginesReceiveRenderedPrompt=false, even
// if the tokenizer delta is within tolerance — compare no longer owns the
// rendering, so the tokenizer divergence might actually come from a prompt
// drift rather than tokenizer-vocabulary noise.
{
  const section = buildCompareSection({
    cacheMode: 'warm',
    loadMode: 'opfs',
    maxTokens: 64,
    promptContract: {
      promptRendered: 'the sky is clear',
      enginesReceiveRenderedPrompt: false,
    },
    prefillTokenTarget: 64,
    doppler: {
      result: {
        result: {
          metrics: {
            avgPrefillTokens: 63,
            avgDecodeTokens: 64,
            generatedText: 'Coherent Doppler output.',
          },
        },
      },
    },
    transformersjs: {
      generatedText: 'Coherent TJS output.',
      runs: [
        {
          prefillTokens: 64,
          decodeTokens: 64,
        },
      ],
    },
  });
  assert.equal(section.pairedComparable, false);
  assert.equal(section.promptTokens.pairedComparable, false);
  assert.equal(section.promptTokens.toleratedTokenizerDelta, null);
}

// Delta > MAX_TOLERATED_TOKENIZER_DELTA must stay non-comparable even with
// every other gate clean.
{
  const promptContract = renderComparePrompt({
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    prompt: 'the sky is clear',
    useChatTemplate: true,
  });
  const section = buildCompareSection({
    cacheMode: 'warm',
    loadMode: 'opfs',
    maxTokens: 64,
    promptContract,
    prefillTokenTarget: 64,
    doppler: {
      result: {
        result: {
          metrics: {
            avgPrefillTokens: 61,
            avgDecodeTokens: 64,
            generatedText: 'Coherent Doppler output.',
          },
        },
      },
    },
    transformersjs: {
      generatedText: 'Coherent TJS output.',
      runs: [
        {
          prefillTokens: 64,
          decodeTokens: 64,
        },
      ],
    },
  });
  assert.equal(section.pairedComparable, false);
  assert.equal(section.invalidReason, 'prompt-token-count-mismatch');
  assert.equal(section.promptTokens.tokenizerDelta, 3);
  assert.equal(section.promptTokens.pairedComparable, false);
  assert.equal(section.promptTokens.toleratedTokenizerDelta, null);
}

{
  const compareConfig = {
    modelProfileById: new Map([
      ['gemma-4-e2b-it-q4k-ehf16-af32', {
        dopplerModelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
        defaultUseChatTemplate: true,
        defaultLoadMode: 'http',
        defaultLoadModeReason: 'unit load-mode reason',
      }],
    ]),
  };
  const compareProfile = resolveCompareProfile(compareConfig, 'gemma-4-e2b-it-q4k-ehf16-af32');
  const sharedContract = buildSharedBenchmarkContract({
    prompt: 'unit prompt',
    maxTokens: 4,
    warmupRuns: 0,
    timedRuns: 1,
    seed: 7,
    sampling: {
      temperature: 0,
      topK: 1,
      topP: 1,
    },
    useChatTemplate: parseCompareOnOff(undefined, compareProfile.defaultUseChatTemplate ?? false, '--use-chat-template'),
  });
  assert.equal(compareProfile.defaultUseChatTemplate, true);
  assert.equal(compareProfile.defaultLoadMode, 'http');
  assert.equal(compareProfile.defaultLoadModeReason, 'unit load-mode reason');
  assert.equal(sharedContract.useChatTemplate, true);
}

{
  const remoteManifest = {
    schemaVersion: 1,
    inference: {
      layerPattern: {
        every: 1,
        offset: null,
      },
      execution: {
        steps: [
          {
            op: 'attn_decode',
            kernel: 'attention_decode_online_f16kv.wgsl',
          },
        ],
      },
    },
  };
  const localManifest = {
    inference: {
      execution: {
        steps: [
          {
            kernel: 'attention_decode_online_f16kv.wgsl',
            op: 'attn_decode',
          },
        ],
      },
      layerPattern: {
        offset: null,
        every: 1,
      },
    },
    schemaVersion: 1,
  };
  const freshness = buildDopplerManifestFreshnessArtifact({
    dopplerModelId: 'unit-model',
    manifestSourceType: 'remote',
    activeManifest: remoteManifest,
    activeManifestSource: 'https://example.test/unit-model/manifest.json',
    activeManifestSha256: 'remote-raw',
    localManifest,
    localManifestSource: path.join(process.cwd(), 'models', 'local', 'unit-model', 'manifest.json'),
    localManifestSha256: 'local-raw',
    enforced: true,
  });
  assert.equal(freshness.checked, true);
  assert.equal(freshness.ok, true);
  assert.equal(freshness.enforced, true);
  assert.equal(freshness.reason, 'remote-matches-local-current-manifest');
  assert.equal(freshness.active.manifestSha256, 'remote-raw');
  assert.equal(freshness.localCurrent.manifestSha256, 'local-raw');
  assert.equal(freshness.active.canonicalManifestSha256, freshness.localCurrent.canonicalManifestSha256);
}

{
  const freshness = buildDopplerManifestFreshnessArtifact({
    dopplerModelId: 'unit-model',
    manifestSourceType: 'remote',
    activeManifest: {
      inference: {
        layerPattern: {
          offset: null,
        },
      },
    },
    activeManifestSource: 'https://example.test/unit-model/manifest.json',
    activeManifestSha256: 'remote-raw',
    localManifest: {
      inference: {
        layerPattern: {
          offset: 5,
        },
      },
    },
    localManifestSource: path.join(process.cwd(), 'models', 'local', 'unit-model', 'manifest.json'),
    localManifestSha256: 'local-raw',
    enforced: true,
  });
  assert.equal(freshness.checked, true);
  assert.equal(freshness.ok, false);
  assert.equal(freshness.enforced, true);
  assert.equal(freshness.reason, 'remote-differs-from-local-current-manifest');
  assert.match(freshness.errors[0], /Remote Doppler manifest for "unit-model" does not match the current local manifest/);
  assert.match(freshness.errors[0], /--allow-non-comparable-lane/);
}

{
  const compareProfile = resolveCompareProfile({
    modelProfileById: new Map([
      ['gemma-4-e2b-it-q4k-ehf16-af32', {
        dopplerModelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
        modelBaseDir: 'local',
      }],
    ]),
  }, 'gemma-4-e2b-it-q4k-ehf16-af32');
  const source = await resolveDopplerModelSource(
    compareProfile,
    'gemma-4-e2b-it-q4k-ehf16-af32',
    null
  );
  assert.equal(source.source, 'local');
  assert.match(source.modelUrl, /^file:\/\//);
  assert.match(source.manifestSource, /models[\\/]+local[\\/]+gemma-4-e2b-it-q4k-ehf16-af32[\\/]manifest\.json$/);
}

{
  const sharedContract = buildSharedBenchmarkContract({
    prompt: 'unit prompt',
    maxTokens: 8,
    warmupRuns: 0,
    timedRuns: 1,
    seed: 7,
    sampling: {
      temperature: 0,
      topK: 1,
      topP: 1,
    },
    useChatTemplate: false,
  });
  const runtimeConfig = buildDopplerRuntimeConfig(sharedContract, {
    batchSize: 4,
    readbackInterval: 4,
    disableMultiTokenDecode: true,
    speculationMode: 'none',
    stopCheckMode: 'per-token',
    kernelPath: null,
    kernelPathPolicy: {
      mode: 'capability-aware',
      sourceScope: ['model', 'manifest', 'config'],
      allowSources: ['model', 'manifest', 'config'],
      onIncompatible: 'remap',
    },
    runtimeBaseConfigJson: {
      inference: {
        batching: {
          batchSize: 99,
          readbackInterval: 99,
        },
        session: {
          decodeLoop: {
            batchSize: 99,
            readbackInterval: 99,
          },
          perLayerInputs: {
            materialization: 'gpu_split_tables',
          },
        },
      },
      shared: {
        bufferPool: {
          budget: {
            maxTotalBytes: 1234,
          },
        },
      },
    },
    runtimeConfigJson: null,
  });
  assert.equal(runtimeConfig.inference.generation.maxTokens, 8);
  assert.equal(runtimeConfig.inference.generation.disableMultiTokenDecode, true);
  assert.deepEqual(runtimeConfig.inference.session.decodeLoop, {
    batchSize: 4,
    stopCheckMode: 'per-token',
    readbackInterval: 4,
    disableCommandBatching: false,
  });
  assert.deepEqual(runtimeConfig.inference.session.speculation, {
    mode: 'none',
    tokens: 1,
    verify: 'greedy',
    threshold: null,
    rollbackOnReject: true,
  });
  assert.equal(runtimeConfig.inference.batching.batchSize, 4);
  assert.equal(runtimeConfig.inference.batching.readbackInterval, 4);
  assert.equal(runtimeConfig.inference.session.decodeLoop.batchSize, 4);
  assert.equal(runtimeConfig.inference.session.decodeLoop.readbackInterval, 4);
  assert.equal(runtimeConfig.inference.session.perLayerInputs.materialization, 'gpu_split_tables');
  assert.equal(runtimeConfig.shared.bufferPool.budget.maxTotalBytes, 1234);
}

{
  const kvCachePlan = resolveDopplerBenchmarkKvCachePlan({
    prefillTokenTarget: 64,
    maxTokens: 8,
    promptContract: {
      enginesReceiveRenderedPrompt: true,
    },
  });
  assert.deepEqual(kvCachePlan, {
    schemaVersion: 1,
    enabled: true,
    source: 'declared-workload-token-budget',
    maxSeqLen: 73,
    promptTokenTarget: 64,
    decodeTokenTarget: 8,
    tokenizerDeltaMargin: 1,
    runtimePaths: [
      'inference.kvcache.maxSeqLen',
      'inference.session.kvcache.maxSeqLen',
    ],
    reason: 'tokenizer-accurate synthetic prompt budget plus decode target and tokenizer tolerance',
  });
  assert.equal(
    resolveDopplerBenchmarkKvCachePlan({
      prefillTokenTarget: null,
      maxTokens: 8,
      promptContract: {
        enginesReceiveRenderedPrompt: true,
      },
    }).reason,
    'prefill-token-target-unavailable'
  );
  assert.equal(
    resolveDopplerBenchmarkKvCachePlan({
      prefillTokenTarget: 64,
      maxTokens: 8,
      promptContract: {
        enginesReceiveRenderedPrompt: false,
      },
    }).reason,
    'prompt-rendering-not-shared'
  );
}

{
  const sharedContract = buildSharedBenchmarkContract({
    prompt: 'unit prompt',
    maxTokens: 8,
    warmupRuns: 0,
    timedRuns: 1,
    seed: 7,
    sampling: {
      temperature: 0,
      topK: 1,
      topP: 1,
    },
    useChatTemplate: false,
  });
  const runtimeConfig = buildDopplerRuntimeConfig(sharedContract, {
    batchSize: 1,
    readbackInterval: 1,
    disableMultiTokenDecode: true,
    speculationMode: 'none',
    stopCheckMode: 'per-token',
    kernelPath: null,
    kernelPathPolicy: null,
    kvCachePlan: resolveDopplerBenchmarkKvCachePlan({
      prefillTokenTarget: 64,
      maxTokens: 8,
      promptContract: {
        enginesReceiveRenderedPrompt: true,
      },
    }),
    runtimeBaseConfigJson: {
      inference: {
        kvcache: {
          maxSeqLen: 4096,
          layout: 'contiguous',
        },
        session: {
          kvcache: {
            maxSeqLen: 4096,
            kvDtype: 'f16',
          },
        },
      },
    },
    runtimeConfigJson: null,
  });
  assert.deepEqual(runtimeConfig.inference.kvcache, {
    maxSeqLen: 73,
    layout: 'contiguous',
  });
  assert.deepEqual(runtimeConfig.inference.session.kvcache, {
    maxSeqLen: 73,
    kvDtype: 'f16',
  });
}

{
  const sharedContract = buildSharedBenchmarkContract({
    prompt: 'unit prompt',
    maxTokens: 8,
    warmupRuns: 0,
    timedRuns: 1,
    seed: 7,
    sampling: {
      temperature: 0,
      topK: 1,
      topP: 1,
    },
    useChatTemplate: false,
  });
  const runtimeConfig = buildDopplerRuntimeConfig(sharedContract, {
    batchSize: 4,
    readbackInterval: 4,
    disableMultiTokenDecode: false,
    speculationMode: 'none',
    stopCheckMode: 'batch',
    kernelPath: null,
    kernelPathPolicy: {
      mode: 'capability-aware',
      sourceScope: ['model', 'manifest', 'config'],
      allowSources: ['model', 'manifest', 'config'],
      onIncompatible: 'remap',
    },
    loadMode: 'http',
    runtimeBaseConfigJson: {
      loading: {
        distribution: {
          sourceOrder: ['cache', 'http'],
        },
        shardCache: {
          verifyHashes: true,
        },
      },
    },
    runtimeLoadModeConfigJson: {
      loading: {
        distribution: {
          sourceOrder: ['http'],
        },
        shardCache: {
          verifyHashes: false,
          rangeCacheBlockBytes: 67108864,
          rangeCacheMaxBytes: 536870912,
          rangeCacheMinBytes: 4096,
          maxConcurrentLoads: 8,
        },
        prefetch: {
          allowRangeLoaderPrefetch: true,
          layersAhead: 1,
          maxShards: 8,
        },
      },
    },
    runtimeConfigJson: null,
  });
  assert.equal(runtimeConfig.shared.benchmark.run.loadMode, 'http');
  assert.deepEqual(runtimeConfig.loading.distribution.sourceOrder, ['http']);
  assert.equal(runtimeConfig.loading.shardCache.verifyHashes, false);
  assert.equal(runtimeConfig.loading.shardCache.rangeCacheBlockBytes, 67108864);
  assert.equal(runtimeConfig.loading.shardCache.rangeCacheMaxBytes, 536870912);
  assert.equal(runtimeConfig.loading.shardCache.rangeCacheMinBytes, 4096);
  assert.equal(runtimeConfig.loading.shardCache.maxConcurrentLoads, 8);
  assert.equal(runtimeConfig.loading.prefetch.allowRangeLoaderPrefetch, true);
  assert.equal(runtimeConfig.loading.prefetch.maxShards, 8);
}

{
  const result = runCompareEngines([
    '--doppler-surface', 'invalid-surface',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
}

{
  const result = runCompareEngines([
    '--doppler-surface', 'bun',
    '--doppler-no-opfs-cache',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /--doppler-no-opfs-cache is browser-only/);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-compare-config-'));
  const bunConfigPath = path.join(tempDir, 'bun-surface-compare-config.json');
  const bunConfig = {
    schemaVersion: 1,
    updated: '2026-06-27',
    defaults: {
      warmLoadMode: 'opfs',
      coldLoadMode: 'http',
    },
    modelProfiles: [
      {
        dopplerModelId: 'unit-bun-surface-model',
        defaultTjsModelId: null,
        defaultDopplerSource: 'local',
        modelBaseDir: 'local',
        defaultDopplerSurface: 'bun',
        defaultDopplerFormat: 'rdrr',
        defaultTjsFormat: null,
        safetensorsSourceId: null,
        compareLane: 'capability_only',
        compareLaneReason: 'unit-test bun surface schema coverage',
      },
    ],
  };
  await fs.writeFile(bunConfigPath, `${JSON.stringify(bunConfig, null, 2)}\n`, 'utf8');
  const result = runCompareEngines([
    '--compare-config', bunConfigPath,
    '--model-id', 'unit-bun-surface-model',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /allow-non-comparable-lane/);
}

{
  const result = runCompareEngines([
    '--mode', 'warm',
    '--doppler-bottleneck-profile', 'on',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /--doppler-bottleneck-profile requires --mode compute or --mode all/);
}

{
  const result = runCompareEngines([
    '--mode', 'compute',
    '--doppler-format', 'rdrr',
    '--load-mode', 'memory',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /RDRR compare lanes do not support --load-mode memory/);
  assertCommandOutputMatches(result, /--doppler-format safetensors/);
}

{
  const result = runCompareEngines([
    '--runtime-config-json',
    '{"inference":{"prompt":"override"}}',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /--runtime-config-json must not override compare-managed fairness or cadence fields/);
}

{
  const result = runCompareEngines([
    '--runtime-config-json',
    '{"inference":{"session":{"speculation":{"mode":"self","tokens":1,"verify":"greedy","threshold":null,"rollbackOnReject":true}}}}',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /--runtime-config-json must not override compare-managed fairness or cadence fields/);
}

{
  const result = runCompareEngines([
    '--runtime-config-json',
    '{"inference":{"kvcache":{"maxSeqLen":4096},"session":{"kvcache":{"maxSeqLen":4096}}}}',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /Doppler KV cache capacity is compare-managed/);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-compare-config-'));
  const badConfigPath = path.join(tempDir, 'bad-compare-config.json');
  const badConfig = {
    schemaVersion: 1,
    updated: '2026-03-05',
    defaults: {
      warmLoadMode: 'opfs',
      coldLoadMode: 'http',
    },
    modelProfiles: [
      {
        dopplerModelId: 'gemma-3-270m-it-f16-af32',
        defaultTjsModelId: 'onnx-community/gemma-3-270m-it-ONNX',
        defaultDopplerSource: 'local',
        modelBaseDir: 'local',
        defaultDopplerSurface: 'unsupported',
        compareLane: 'performance_comparable',
      },
    ],
  };
  await fs.writeFile(badConfigPath, `${JSON.stringify(badConfig, null, 2)}\n`, 'utf8');
  const result = runCompareEngines([
    '--compare-config', badConfigPath,
    '--json',
  ]);
  assert.notEqual(result.status, 0);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-compare-config-'));
  const badConfigPath = path.join(tempDir, 'bad-compare-config-runtime-profile.json');
  const badConfig = {
    schemaVersion: 1,
    updated: '2026-04-10',
    defaults: {
      warmLoadMode: 'opfs',
      coldLoadMode: 'http',
    },
    modelProfiles: [
      {
        dopplerModelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
        defaultTjsModelId: 'onnx-community/gemma-4-E2B-it-ONNX',
        defaultDopplerSource: 'local',
        modelBaseDir: 'local',
        defaultDopplerSurface: 'browser',
        compareLane: 'performance_comparable',
        dopplerRuntimeProfileByDecodeProfile: {
          throughput: true,
        },
      },
    ],
  };
  await fs.writeFile(badConfigPath, `${JSON.stringify(badConfig, null, 2)}\n`, 'utf8');
  const result = runCompareEngines([
    '--compare-config', badConfigPath,
    '--model-id', 'gemma-4-e2b-it-q4k-ehf16-af32',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /dopplerRuntimeProfileByDecodeProfile/i);
  assertCommandOutputMatches(result, /string\s+\|\s+null/i);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-compare-config-'));
  const badConfigPath = path.join(tempDir, 'bad-compare-config-chat-template.json');
  const badConfig = {
    schemaVersion: 1,
    updated: '2026-04-09',
    defaults: {
      warmLoadMode: 'opfs',
      coldLoadMode: 'http',
    },
    modelProfiles: [
      {
        dopplerModelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
        defaultTjsModelId: 'onnx-community/gemma-4-E2B-it-ONNX',
        defaultDopplerSource: 'local',
        modelBaseDir: 'local',
        defaultDopplerSurface: 'browser',
        defaultUseChatTemplate: 'yes',
        compareLane: 'capability_only',
        compareLaneReason: 'unit-test invalid chat template default',
      },
    ],
  };
  await fs.writeFile(badConfigPath, `${JSON.stringify(badConfig, null, 2)}\n`, 'utf8');
  const result = runCompareEngines([
    '--compare-config', badConfigPath,
    '--model-id', 'gemma-4-e2b-it-q4k-ehf16-af32',
    '--allow-non-comparable-lane',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /defaultUseChatTemplate/i);
  assertCommandOutputMatches(result, /boolean(?:\s+\|\s+null| or null)/i);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-compare-config-'));
  const badConfigPath = path.join(tempDir, 'bad-compare-config-load-mode-reason.json');
  const badConfig = {
    schemaVersion: 1,
    updated: '2026-04-21',
    defaults: {
      warmLoadMode: 'opfs',
      coldLoadMode: 'http',
    },
    modelProfiles: [
      {
        dopplerModelId: 'qwen-3-5-2b-q4k-ehaf16',
        defaultTjsModelId: 'onnx-community/Qwen3.5-2B-ONNX',
        defaultDopplerSource: 'quickstart-registry',
        modelBaseDir: null,
        defaultDopplerSurface: 'browser',
        defaultLoadMode: 'http',
        compareLane: 'performance_comparable',
      },
    ],
  };
  await fs.writeFile(badConfigPath, `${JSON.stringify(badConfig, null, 2)}\n`, 'utf8');
  const result = runCompareEngines([
    '--compare-config', badConfigPath,
    '--model-id', 'qwen-3-5-2b-q4k-ehaf16',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /defaultLoadModeReason/);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-compare-config-'));
  const nonComparableConfigPath = path.join(tempDir, 'capability-only-compare-config.json');
  const nonComparableConfig = {
    schemaVersion: 1,
    updated: '2026-03-27',
    defaults: {
      warmLoadMode: 'opfs',
      coldLoadMode: 'http',
    },
    modelProfiles: [
      {
        dopplerModelId: 'unit-capability-model',
        defaultTjsModelId: null,
        defaultDopplerSource: 'local',
        modelBaseDir: 'local',
        defaultDopplerSurface: 'auto',
        defaultDopplerFormat: 'rdrr',
        defaultTjsFormat: null,
        safetensorsSourceId: null,
        compareLane: 'capability_only',
        compareLaneReason: 'unit-test support-only lane',
      },
    ],
  };
  await fs.writeFile(nonComparableConfigPath, `${JSON.stringify(nonComparableConfig, null, 2)}\n`, 'utf8');
  const result = runCompareEngines([
    '--compare-config', nonComparableConfigPath,
    '--model-id', 'unit-capability-model',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /allow-non-comparable-lane/);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-compare-config-'));
  const missingDefaultsPath = path.join(tempDir, 'missing-defaults-compare-config.json');
  const payload = {
    schemaVersion: 1,
    updated: '2026-04-10',
    modelProfiles: [
      {
        dopplerModelId: 'gemma-3-270m-it-f16-af32',
        defaultTjsModelId: 'onnx-community/gemma-3-270m-it-ONNX',
        defaultDopplerSource: 'local',
        modelBaseDir: 'local',
        defaultDopplerSurface: 'auto',
        compareLane: 'performance_comparable',
      },
    ],
  };
  await fs.writeFile(missingDefaultsPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  const result = runCompareEngines([
    '--compare-config', missingDefaultsPath,
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /defaults/i);
}

console.log('compare-engines-cli-contract.test: ok');
