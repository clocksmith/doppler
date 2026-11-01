import { buildComparisonFingerprint, resolveObservationPolicy } from '../../client/inspection.js';
import { buildGenerationBackendIdentity } from '../../client/model-host/model-evidence.js';
import { isExecutionObservationRequested } from '../execution-cost-ledger.js';
import { resolveAutomaticGenerationDiagnostics } from './text-input.js';
import { hasNodeVersion } from '../../storage/runtime-env.js';
import { shouldCaptureReferenceLogits, shouldCaptureReferenceKvBytes,
  shouldEnableReferenceTranscriptDiagnostics } from './text-evidence.js';

/** @returns {import('./benchmark-comparison.js').BenchmarkEnvironment} */
export function resolveBenchmarkEnvironment() {
  return {
    library: 'doppler', runtime: hasNodeVersion() ? 'node' : 'browser', device: 'webgpu',
    browserUserAgent: typeof navigator !== 'undefined' ? (navigator.userAgent || null) : null,
    browserPlatform: typeof navigator !== 'undefined' ? (navigator.platform || null) : null,
    browserLanguage: typeof navigator !== 'undefined' ? (navigator.language || null) : null,
    browserVendor: typeof navigator !== 'undefined' ? (navigator.vendor || null) : null,
  };
}

/** @type {import('./benchmark-comparison.js').CaptureBenchmarkComparison} */
// Only the uninstrumented generation lane implements the registry's coarse
// observation contract. Missing evidence is reported, never synthesized.
const captureBenchmarkComparison = function(harness, run, runtimeConfig, runOptions, deviceInfo) {
  try {
    if (resolveAutomaticGenerationDiagnostics(runtimeConfig, runOptions)?.enabled === true
      || isExecutionObservationRequested(runtimeConfig)
      || runtimeConfig.shared?.harness?.mode === 'diagnose'
      || (runtimeConfig.shared?.debug?.probes?.length ?? 0) > 0
      || shouldEnableReferenceTranscriptDiagnostics(runOptions, runtimeConfig)
      || shouldCaptureReferenceLogits(runOptions, runtimeConfig)
      || shouldCaptureReferenceKvBytes(runOptions, runtimeConfig)) {
      throw new Error('Instrumented benchmark does not implement a representative observation policy.');
    }
    if (!harness.manifestHash) {
      throw new Error('Original manifest byte identity is unavailable.');
    }
    if (!run?.generationConfig) {
      throw new Error('Actual generation sampling configuration is unavailable.');
    }
    if (!run.initialExecutionIdentity?.digest) {
      throw new Error('Resolved initial execution identity is unavailable.');
    }
    if (!deviceInfo?.adapterInfo) {
      throw new Error('Actual WebGPU adapter identity is unavailable.');
    }
    const execution = buildGenerationBackendIdentity({
      kernelCapabilities: deviceInfo,
      stats: harness.pipeline.getStats?.(),
    });
    if (!Object.values(execution.adapter).some((value) => value && value !== 'unknown')) {
      throw new Error('Actual WebGPU adapter identity is unavailable.');
    }
    if (!execution.executionPlanId || !execution.activationDtype) {
      throw new Error('Resolved execution plan identity is unavailable.');
    }
    if (!harness.manifest?.modelId || !run.promptTokenIds) {
      throw new Error('Artifact model or actual prompt token identity is unavailable.');
    }
    const comparisonFingerprint = buildComparisonFingerprint({
      artifact: { modelId: harness.manifest?.modelId, manifestHash: harness.manifestHash },
      tokenizer: harness.manifest?.tokenizer,
      promptTokenIds: run.promptTokenIds,
      sampling: run.generationConfig,
      observationPolicyId: resolveObservationPolicy().id,
      execution: { ...execution, initialExecutionIdentityDigest: run.initialExecutionIdentity.digest },
      browser: typeof navigator === 'undefined' ? {} : {
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        language: navigator.language,
      },
      adapter: execution.adapter,
    });
    return { comparisonFingerprint, comparisonFingerprintUnavailableReason: null };
  } catch (error) {
    return {
      comparisonFingerprint: null,
      comparisonFingerprintUnavailableReason: error instanceof Error ? error.message : String(error),
    };
  }
};

/** @type {import('./benchmark-comparison.js').CreateBenchmarkComparisonCapture} */
export function createBenchmarkComparisonCapture(harness, runtimeConfig, runOptions, deviceInfo) {
  let captured = false;
  return {
    result: /** @type {import('./benchmark-comparison.js').BenchmarkComparison} */ ({
      comparisonFingerprint: null,
      comparisonFingerprintUnavailableReason: 'Comparison evidence is unavailable for this workload.',
    }),
    capture(run) {
      const next = captureBenchmarkComparison(harness, run, runtimeConfig, runOptions, deviceInfo);
      if (!captured) {
        this.result = next;
        captured = true;
      } else if (this.result.comparisonFingerprint
        && next.comparisonFingerprint?.fullDigest !== this.result.comparisonFingerprint.fullDigest) {
        this.result = {
          comparisonFingerprint: null,
          comparisonFingerprintUnavailableReason: next.comparisonFingerprintUnavailableReason
            ?? 'Comparison identity changed between timed runs.',
        };
      }
    },
  };
}
