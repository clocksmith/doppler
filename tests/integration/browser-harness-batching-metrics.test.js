import assert from 'node:assert/strict';

const { runBrowserSuite } = await import('../../src/inference/browser-harness.js');
const {
  buildDecodeRecordTopOpGroups,
  runGeneration,
} = await import('../../src/inference/browser-harness-text-helpers.js');
const { createDopplerConfig } = await import('../../src/config/schema/doppler.schema.js');
const { getRuntimeConfig, setRuntimeConfig } = await import('../../src/config/runtime.js');

{
  const pipeline = {
    async *generate(_promptInput, options = {}) {
      options.onToken?.(1, 'Blue');
      yield 'Blue';
    },
    getStats() {
      return {
        prefillTimeMs: 2,
        ttftMs: 3,
        decodeTimeMs: 4,
        prefillTokens: 5,
        decodeTokens: 6,
        decodeMode: 'batched_gpu_stepwise_ple',
        batchGuardReason: null,
        singleTokenReadbackWaitMs: 7,
        singleTokenReadbackMapWaitMs: 1,
        singleTokenReadbackCleanupMs: 2,
        singleTokenReadbackCopyMs: 3,
        singleTokenOrchestrationMs: 8,
        decodeRecordMs: 1,
        decodeRecordOps: 5,
        decodeRecordPasses: 2,
        decodeRecordOpLabels: {
          matmul: 3,
          sample: 2,
        },
        uniformCache: {
          hits: 8,
          misses: 2,
          hitRate: '80.0%',
          evictions: 1,
          currentSize: 4,
          pendingDestruction: 0,
        },
        decodeSubmitWaitMs: 1,
        decodeReadbackWaitMs: 1,
        decodeReadbackMapWaitMs: 0.4,
        decodeReadbackCleanupMs: 0.3,
        decodeReadbackCopyMs: 0.2,
        batching: {
          batchedForwardCalls: 9,
          unbatchedForwardCalls: 10,
          totalBatchedTimeMs: 11,
          totalUnbatchedTimeMs: 12,
          gpuSubmissions: 13,
          requestedBatchTokens: 16,
          effectiveBatchTokens: 4,
          executedBatchTokens: 32,
          resolvedBatchTokens: 15,
          maxBatchTokenCap: 4,
          batchClampCount: 3,
        },
        decodeProfileSteps: [],
      };
    },
  };

  const result = await runGeneration(pipeline, {
    inference: {
      prompt: 'The sky is',
      generation: {
        maxTokens: 1,
      },
      sampling: {
        temperature: 0,
        topK: 1,
        topP: 1,
      },
    },
    shared: {
      tooling: {
        intent: 'calibrate',
      },
    },
  });

  assert.equal(result.phase.decodeMode, 'batched_gpu_stepwise_ple');
  assert.equal(result.phase.batching?.batchedForwardCalls, 9);
  assert.equal(result.phase.batching?.unbatchedForwardCalls, 10);
  assert.equal(result.phase.batching?.totalBatchedTimeMs, 11);
  assert.equal(result.phase.batching?.totalUnbatchedTimeMs, 12);
  assert.equal(result.phase.batching?.gpuSubmissions, 13);
  assert.equal(result.phase.batching?.requestedBatchTokens, 16);
  assert.equal(result.phase.batching?.effectiveBatchTokens, 4);
  assert.equal(result.phase.batching?.executedBatchTokens, 32);
  assert.equal(result.phase.batching?.resolvedBatchTokens, 15);
  assert.equal(result.phase.batching?.maxBatchTokenCap, 4);
  assert.equal(result.phase.batching?.batchClampCount, 3);
  assert.equal(result.phase.gpu?.singleTokenReadbackWaitMs, 7);
  assert.equal(result.phase.gpu?.singleTokenReadbackMapWaitMs, 1);
  assert.equal(result.phase.gpu?.singleTokenReadbackCleanupMs, 2);
  assert.equal(result.phase.gpu?.singleTokenReadbackCopyMs, 3);
  assert.equal(result.phase.gpu?.singleTokenOrchestrationMs, 8);
  assert.equal(result.phase.gpu?.decodeReadbackMapWaitMs, 0.4);
  assert.equal(result.phase.gpu?.decodeReadbackCleanupMs, 0.3);
  assert.equal(result.phase.gpu?.decodeReadbackCopyMs, 0.2);
  assert.equal(result.phase.gpu?.decodeRecordOps, 5);
  assert.equal(result.phase.gpu?.decodeRecordPasses, 2);
  assert.equal(result.phase.gpu?.decodeRecordUniqueOpLabels, 2);
  assert.deepEqual(result.phase.gpu?.decodeRecordTopOps, [
    { label: 'matmul', count: 3, shareOfOps: 0.6 },
    { label: 'sample', count: 2, shareOfOps: 0.4 },
  ]);
  assert.deepEqual(result.phase.gpu?.decodeRecordTopOpGroups, [
    { label: 'matmul', count: 3, shareOfOps: 0.6 },
    { label: 'sample', count: 2, shareOfOps: 0.4 },
  ]);
  assert.deepEqual(result.phase.gpu?.uniformCache, {
    hits: 8,
    misses: 2,
    totalLookups: 10,
    hitRateRatio: 0.8,
    hitRate: '80.0%',
    evictions: 1,
    currentSize: 4,
    pendingDestruction: 0,
  });
  assert.equal(result.phase.gpu?.decodeRecordMsPerOp, 0.2);
  assert.equal(result.phase.gpu?.decodeRecordMsPerPass, 0.5);
  assert.equal(result.phase.gpu?.decodeRecordPassesPerOp, 0.4);
  assert.equal(result.phase.gpu?.decodeRecordMsPerExecutedBatchToken, 0.03125);
  assert.equal(result.phase.gpu?.decodeRecordOpsPerExecutedBatchToken, 0.15625);
  assert.equal(result.phase.gpu?.decodeRecordPassesPerExecutedBatchToken, 0.0625);
  assert.equal(result.phase.gpu?.decodeOrchestrationMs, 2);
}

{
  const previousRuntimeConfig = getRuntimeConfig();
  setRuntimeConfig(createDopplerConfig({
    runtime: {
      inference: {
        generation: {
          disableMultiTokenDecode: false,
        },
        batching: {
          batchSize: 8,
          readbackInterval: 4,
          stopCheckMode: 'batch',
          readbackMode: 'overlapped',
        },
        session: {
          decodeLoop: {
            batchSize: 8,
            readbackInterval: 4,
            stopCheckMode: 'batch',
            readbackMode: 'overlapped',
            ringTokens: 2,
            ringStop: 1,
            ringStaging: 2,
            disableCommandBatching: false,
          },
        },
      },
    },
  }).runtime);
  let result;
  try {
    result = await runBrowserSuite({
      suite: 'bench',
      command: 'bench',
      surface: 'node',
      harnessOverride: {
        modelLoadMs: 1,
        manifest: {
          modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
          modelType: 'transformer',
          architecture: {
            numLayers: 42,
            hiddenSize: 2048,
            intermediateSize: 16384,
            numAttentionHeads: 8,
            numKeyValueHeads: 1,
            headDim: 256,
            vocabSize: 262144,
            maxSeqLen: 32768,
          },
          inference: {
            attention: {
              queryPreAttnScalar: 256,
            },
            chatTemplate: {
              type: 'gemma',
              enabled: true,
            },
          },
        },
        pipeline: {
          async *generate(_promptInput, options = {}) {
            options.onToken?.(1, 'Blue');
            yield 'Blue';
          },
          getStats() {
            return {
              totalTimeMs: 15,
              loadTiming: {
                schemaVersion: 1,
                source: 'doppler-loader',
                modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
                status: 'complete',
                customShardLoader: true,
                byteAccountingMode: 'custom-loader-progress-unavailable',
                totalBytes: 1024,
                totalShards: 2,
                bytesLoaded: 1024,
                shardsLoaded: 2,
                bytesPerSecond: 2048,
                phasesMs: {
                  preflight: 0.1,
                  tensorLocations: 0.2,
                  embeddings: 0.3,
                  layers: 0.4,
                  finalWeights: 0.5,
                  cleanup: 0.1,
                },
                layers: {
                  count: 2,
                  totalMs: 0.4,
                  meanMs: 0.2,
                  maxMs: 0.3,
                  maxLayer: 1,
                },
                totalMs: 0.9,
                failedPhase: null,
                error: null,
              },
              pipelineLoadTiming: {
                schemaVersion: 1,
                source: 'doppler-pipeline',
                modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
                status: 'complete',
                phasesMs: {
                  reset: 0.01,
                  configResolution: 0.02,
                  kernelWarmup: 0.03,
                  tokenizer: 0.04,
                  executionSetup: 0.05,
                  loadWeights: 0.9,
                  rope: 0.06,
                  convStates: 0.07,
                },
                details: {
                  tokenizer: {
                    schemaVersion: 1,
                    source: 'doppler-tokenizer',
                    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
                    status: 'complete',
                    tokenizerType: 'bundled',
                    tokenizerFile: 'tokenizer.json',
                    backend: 'bundled',
                    assetSource: 'custom-loader',
                    cacheHit: false,
                    phasesMs: {
                      configResolution: 0.01,
                      cacheLookup: 0.02,
                      backendCreate: 0.03,
                      assetLoad: 0.04,
                      assetParse: 0.05,
                      backendLoad: 0.06,
                      cacheStore: 0.07,
                    },
                    totalMs: 0.28,
                    error: null,
                  },
                },
                totalMs: 0.95,
              },
              prefillTimeMs: 5,
              ttftMs: 6,
              decodeTimeMs: 10,
              prefillTokens: 7,
              decodeTokens: 1,
              decodeMode: 'batched_gpu_stepwise_ple',
              batchGuardReason: null,
              singleTokenReadbackWaitMs: 4,
              singleTokenReadbackMapWaitMs: 1,
              singleTokenReadbackCleanupMs: 2,
              singleTokenReadbackCopyMs: 3,
              singleTokenOrchestrationMs: 2,
              decodeRecordMs: 3,
              decodeRecordOps: 12,
              decodeRecordPasses: 1,
              decodeRecordOpLabels: {
                'matmul:ffn_down:L0': 4,
                'matmul:ffn_down:L1': 5,
                attention: 3,
              },
              uniformCache: {
                hits: 30,
                misses: 6,
                hitRate: '83.3%',
                evictions: 2,
                currentSize: 12,
                pendingDestruction: 1,
              },
              decodeSubmitWaitMs: 2,
              decodeReadbackWaitMs: 2,
              decodeReadbackMapWaitMs: 0.8,
              decodeReadbackCleanupMs: 0.4,
              decodeReadbackCopyMs: 0.2,
              batching: {
                batchedForwardCalls: 3,
                unbatchedForwardCalls: 0,
                totalBatchedTimeMs: 10,
                totalUnbatchedTimeMs: 0,
                gpuSubmissions: 1,
                requestedBatchTokens: 16,
                effectiveBatchTokens: 4,
                executedBatchTokens: 8,
                resolvedBatchTokens: 3,
                maxBatchTokenCap: 4,
                batchClampCount: 1,
              },
              executionPlan: {
                primary: {
                  id: 'resolved-plan',
                  kernelPathId: 'test-kernel-path',
                  kernelPathSource: 'manifest',
                  activationDtype: 'f16',
                  batchSize: 8,
                  readbackInterval: 4,
                  stopCheckMode: 'batch',
                  readbackMode: 'sequential',
                  disableCommandBatching: false,
                  ringTokens: 2,
                  ringStop: 1,
                  ringStaging: 2,
                },
                fallback: null,
                activePlanIdAtStart: 'resolved-plan',
                finalActivePlanId: 'resolved-plan',
                transitions: [],
              },
              decodeProfileSteps: [],
            };
          },
          reset() {},
          async unload() {},
        },
      },
    });
  } finally {
    setRuntimeConfig(previousRuntimeConfig);
  }

  assert.equal(result.results[0]?.passed, true);
  assert.equal(result.metrics.decodeMode, 'batched_gpu_stepwise_ple');
  assert.deepEqual(result.metrics.decodeCadence, {
    batchSize: 8,
    readbackInterval: 4,
    stopCheckMode: 'batch',
    readbackMode: 'sequential',
    disableCommandBatching: false,
    disableMultiTokenDecode: false,
    speculationMode: null,
    tokensPerReadback: 32,
    runtimeMirror: {
      batching: {
        batchSize: 8,
        readbackInterval: 4,
        stopCheckMode: 'batch',
        readbackMode: 'overlapped',
      },
      decodeLoop: {
        batchSize: 8,
        readbackInterval: 4,
        stopCheckMode: 'batch',
        readbackMode: 'overlapped',
        ringTokens: 2,
        ringStop: 1,
        ringStaging: 2,
      },
    },
    executionPlan: {
      id: 'resolved-plan',
      batchSize: 8,
      readbackInterval: 4,
      stopCheckMode: 'batch',
      readbackMode: 'sequential',
      disableCommandBatching: false,
      ringTokens: 2,
      ringStop: 1,
      ringStaging: 2,
    },
  });
  assert.equal(result.metrics.executionPlan?.primary?.readbackMode, 'sequential');
  assert.equal(result.metrics.batching?.batchedForwardCalls?.median, 3);
  assert.equal(result.metrics.batching?.totalBatchedTimeMs?.median, 10);
  assert.equal(result.metrics.batching?.gpuSubmissions?.median, 1);
  assert.equal(result.metrics.batching?.requestedBatchTokens?.median, 16);
  assert.equal(result.metrics.batching?.effectiveBatchTokens?.median, 4);
  assert.equal(result.metrics.batching?.executedBatchTokens?.median, 8);
  assert.equal(result.metrics.batching?.resolvedBatchTokens?.median, 3);
  assert.equal(result.metrics.batching?.maxBatchTokenCap?.median, 4);
  assert.equal(result.metrics.batching?.batchClampCount?.median, 1);
  assert.equal(result.metrics.load?.modelLoadMs, 1);
  assert.equal(result.metrics.load?.loader?.totalMs, 0.9);
  assert.equal(result.metrics.load?.loader?.byteAccountingMode, 'custom-loader-progress-unavailable');
  assert.equal(result.metrics.load?.loader?.phasesMs?.layers, 0.4);
  assert.equal(result.metrics.load?.loader?.layers?.maxLayer, 1);
  assert.equal(result.metrics.load?.pipeline?.totalMs, 0.95);
  assert.equal(result.metrics.load?.pipeline?.phasesMs?.tokenizer, 0.04);
  assert.equal(result.metrics.load?.pipeline?.details?.tokenizer?.backend, 'bundled');
  assert.equal(result.metrics.load?.pipeline?.details?.tokenizer?.assetSource, 'custom-loader');
  assert.equal(result.metrics.load?.pipeline?.details?.tokenizer?.phasesMs?.backendLoad, 0.06);
  assert.equal(result.metrics.load?.residualsMs?.modelLoadMinusLoaderMs, 0.1);
  assert.equal(result.metrics.load?.residualsMs?.modelLoadMinusPipelineMs, 0.05);
  assert.equal(result.metrics.load?.residualsMs?.pipelineMinusLoaderMs, 0.05);
  assert.equal(result.timingDiagnostics.load?.loader?.totalBytes, 1024);
  assert.equal(result.timingDiagnostics.load?.pipeline?.source, 'doppler-pipeline');
  assert.equal(result.timingDiagnostics.load?.pipeline?.details?.tokenizer?.totalMs, 0.28);
  assert.equal(result.timingDiagnostics.load?.consistent?.loaderWithinModelLoad, true);
  assert.equal(result.timingDiagnostics.load?.consistent?.pipelineWithinModelLoad, true);
  assert.equal(result.timingDiagnostics.load?.consistent?.loaderWithinPipeline, true);
  assert.equal(result.metrics.gpu?.singleTokenReadbackWaitMs?.median, 4);
  assert.equal(result.metrics.gpu?.singleTokenReadbackMapWaitMs?.median, 1);
  assert.equal(result.metrics.gpu?.singleTokenReadbackCleanupMs?.median, 2);
  assert.equal(result.metrics.gpu?.singleTokenReadbackCopyMs?.median, 3);
  assert.equal(result.metrics.gpu?.singleTokenOrchestrationMs?.median, 2);
  assert.equal(result.metrics.gpu?.decodeReadbackMapWaitMs?.median, 0.8);
  assert.equal(result.metrics.gpu?.decodeReadbackCleanupMs?.median, 0.4);
  assert.equal(result.metrics.gpu?.decodeReadbackCopyMs?.median, 0.2);
  assert.equal(result.metrics.gpu?.decodeRecordOps?.median, 12);
  assert.equal(result.metrics.gpu?.decodeRecordPasses?.median, 1);
  assert.equal(result.metrics.gpu?.decodeRecordUniqueOpLabels, 3);
  assert.deepEqual(result.metrics.gpu?.decodeRecordTopOps, [
    { label: 'matmul:ffn_down:L1', count: 5, shareOfOps: 5 / 12 },
    { label: 'matmul:ffn_down:L0', count: 4, shareOfOps: 1 / 3 },
    { label: 'attention', count: 3, shareOfOps: 0.25 },
  ]);
  assert.deepEqual(result.metrics.gpu?.decodeRecordTopOpGroups, [
    { label: 'matmul:ffn_down', count: 9, shareOfOps: 0.75 },
    { label: 'attention', count: 3, shareOfOps: 0.25 },
  ]);
  assert.deepEqual(result.metrics.gpu?.uniformCache, {
    hits: 30,
    misses: 6,
    totalLookups: 36,
    hitRateRatio: 30 / 36,
    hitRate: '83.3%',
    evictions: 2,
    currentSize: 12,
    pendingDestruction: 1,
  });
  assert.equal(result.metrics.gpu?.decodeRecordMsPerOp?.median, 0.25);
  assert.equal(result.metrics.gpu?.decodeRecordMsPerPass?.median, 3);
  assert.equal(result.metrics.gpu?.decodeRecordPassesPerOp?.median, 1 / 12);
  assert.equal(result.metrics.gpu?.decodeRecordMsPerExecutedBatchToken?.median, 0.375);
  assert.equal(result.metrics.gpu?.decodeRecordOpsPerExecutedBatchToken?.median, 1.5);
  assert.equal(result.metrics.gpu?.decodeRecordPassesPerExecutedBatchToken?.median, 0.125);
  assert.equal(result.metrics.gpu?.decodeOrchestrationMs?.median, 5);
  assert.equal(result.metrics.decodeBottleneck?.schemaVersion, 1);
  assert.equal(result.metrics.decodeBottleneck?.dominant?.id, 'orchestration');
  assert.equal(result.metrics.decodeBottleneck?.dominant?.ms, 5);
  assert.equal(result.metrics.decodeBottleneck?.dominant?.shareOfDecode, 0.5);
  assert.equal(result.metrics.decodeBottleneck?.bottleneckClass, 'orchestration');
  assert.equal(result.metrics.decodeBottleneck?.decodeWallMs, 10);
  assert.equal(result.metrics.decodeBottleneck?.componentsMs?.commandRecordMs, 3);
  assert.equal(result.metrics.decodeBottleneck?.componentsMs?.effectiveSubmitReadbackWaitMs, 2);
  assert.equal(result.metrics.decodeBottleneck?.componentsMs?.readbackMapWaitMs, 0.8);
  assert.equal(result.metrics.decodeBottleneck?.componentsMs?.readbackUnattributedMs, 0.6);
  assert.equal(result.metrics.decodeBottleneck?.componentsMs?.gpuTimestampMs, null);
  assert.equal(result.metrics.decodeBottleneck?.componentsMs?.orchestrationMs, 5);
  assert.equal(result.metrics.decodeBottleneck?.shares?.commandRecord, 0.3);
  assert.equal(result.metrics.decodeBottleneck?.shares?.gpuTimestamp, null);
  assert.equal(result.metrics.decodeBottleneck?.recording?.opsPerExecutedBatchToken, 1.5);
  assert.deepEqual(result.metrics.decodeBottleneck?.recording?.topOps, [
    { label: 'matmul:ffn_down:L1', count: 5, shareOfOps: 5 / 12 },
    { label: 'matmul:ffn_down:L0', count: 4, shareOfOps: 1 / 3 },
    { label: 'attention', count: 3, shareOfOps: 0.25 },
  ]);
  assert.deepEqual(result.metrics.decodeBottleneck?.recording?.topOpGroups, [
    { label: 'matmul:ffn_down', count: 9, shareOfOps: 0.75 },
    { label: 'attention', count: 3, shareOfOps: 0.25 },
  ]);
  assert.deepEqual(result.metrics.decodeBottleneck?.recording?.uniformCache, {
    hits: 30,
    misses: 6,
    totalLookups: 36,
    hitRateRatio: 30 / 36,
    evictions: 2,
    currentSize: 12,
    pendingDestruction: 1,
    hitRate: '83.3%',
  });
  assert.equal(result.timingDiagnostics.decodeBottleneck?.dominant?.id, 'orchestration');
  assert.equal(result.timingDiagnostics.decodeBottleneck?.recording?.passCount, 1);
  assert.equal(result.timingDiagnostics.decodeBottleneck?.recording?.uniformCache?.hitRate, '83.3%');
  assert.deepEqual(result.metrics.referenceTranscript?.tokens?.ids, [1]);
  assert.equal(result.metrics.referenceTranscript?.output?.tokensGenerated, 1);
}

{
  assert.deepEqual(
    buildDecodeRecordTopOpGroups(
      {
        'L0.rmsnorm_pair': 2,
        'L1.rmsnorm_pair': 3,
        'matmul:ffn_down:L0': 4,
        'matmul:ffn_down:L1': 5,
      },
      14
    ),
    [
      { label: 'matmul:ffn_down', count: 9, shareOfOps: 9 / 14 },
      { label: 'rmsnorm_pair', count: 5, shareOfOps: 5 / 14 },
    ]
  );
}

{
  // decodeSubmitWaitMs and decodeReadbackWaitMs observe the same submitted GPU work
  // from different points in the command lifecycle, so only the larger wait belongs
  // in the residual.
  const pipeline = {
    async *generate(_promptInput, options = {}) {
      options.onToken?.(1, 'Blue');
      yield 'Blue';
    },
    getStats() {
      return {
        prefillTimeMs: 2,
        ttftMs: 3,
        decodeTimeMs: 5,
        prefillTokens: 1,
        decodeTokens: 1,
        decodeRecordMs: 3,
        decodeSubmitWaitMs: 2,
        decodeReadbackWaitMs: 2,
        decodeProfileSteps: [],
      };
    },
  };

  const result = await runGeneration(pipeline, {
    inference: {
      prompt: 'The sky is',
      generation: { maxTokens: 1 },
      sampling: { temperature: 0, topK: 1, topP: 1 },
    },
    shared: { tooling: { intent: 'calibrate' } },
  });

  assert.equal(result.phase.gpu?.decodeOrchestrationMs, 0);
}

{
  const previousRuntimeConfig = getRuntimeConfig();
  setRuntimeConfig(createDopplerConfig({
    runtime: {
      shared: {
        tooling: {
          intent: 'investigate',
        },
        debug: {
          profiler: {
            enabled: true,
          },
        },
      },
    },
  }).runtime);
  let result;
  try {
    result = await runBrowserSuite({
      suite: 'debug',
      command: 'debug',
      surface: 'node',
      harnessOverride: {
        modelLoadMs: 1,
        manifest: {
          modelId: 'single-run-debug-model',
          modelType: 'transformer',
          inference: {
            chatTemplate: {
              type: 'gemma',
              enabled: true,
            },
          },
        },
        pipeline: {
          tokenizer: {
            decode(ids) {
              return ids.map((id) => String(id)).join('');
            },
          },
          async *generate(_promptInput, options = {}) {
            options.onToken?.(1, 'Blue');
            yield 'Blue';
          },
          getStats() {
            return {
              prefillTimeMs: 2,
              ttftMs: 3,
              decodeTimeMs: 10,
              gpuTimeDecodeMs: 1,
              prefillTokens: 1,
              decodeTokens: 1,
              decodeRecordMs: 2,
              decodeSubmitWaitMs: 3,
              decodeReadbackWaitMs: 3,
              decodeReadbackMapWaitMs: 1,
              decodeReadbackCleanupMs: 0.5,
              decodeReadbackCopyMs: 0.25,
              decodeProfileSteps: [
                {
                  step: 1,
                  timings: {
                    kernel: 1,
                  },
                  totalMs: 1,
                },
              ],
            };
          },
          reset() {},
          async unload() {},
        },
      },
    });
  } finally {
    setRuntimeConfig(previousRuntimeConfig);
  }

  assert.equal(result.results[0]?.passed, true);
  assert.equal(result.metrics.decodeBottleneck?.schemaVersion, 1);
  assert.equal(result.metrics.decodeBottleneck?.componentsMs?.gpuTimestampMs, 1);
  assert.equal(result.metrics.decodeBottleneck?.componentsMs?.submitReadbackSlackMs, 2);
  assert.equal(result.metrics.decodeBottleneck?.shares?.gpuTimestamp, 0.1);
  assert.equal(result.timingDiagnostics.decodeBottleneck?.componentsMs?.gpuTimestampMs, 1);
}

console.log('browser-harness-batching-metrics.test: ok');
