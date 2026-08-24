import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  verifyClosedProgramBundle,
} from './program-bundle.js';
import { validateProgramBundle } from '../config/schema/program-bundle.schema.js';
import {
  bootstrapNodeWebGPU,
  releaseNodeWebGPU,
} from './node-webgpu.js';
import {
  buildDeterministicTokenEvidenceFromReferenceTranscript,
} from './boundary-evidence.js';
import { sha256Hex } from '../formats/sha256.js';
import { stableSortObject } from '../formats/stable-sort-object.js';
import { runNodeCommandExecution } from './node-command-execution.js';

export const PROGRAM_BUNDLE_PARITY_SCHEMA_ID = 'doppler.program-bundle-parity/v2';

const DOE_NATIVE_PROVIDER_OPTIONS = Object.freeze({
  providers: [Object.freeze({
    id: 'doe-native',
    kind: 'module',
    module: 'doe-gpu',
    gpu: Object.freeze({
      kind: 'factory',
      path: 'createNativeDirect',
      args: [Object.freeze(['enable-dawn-features=allow_unsafe_apis'])],
    }),
    globals: Object.freeze({
      GPUBufferUsage: 'globals.GPUBufferUsage',
      GPUShaderStage: 'globals.GPUShaderStage',
      GPUMapMode: 'globals.GPUMapMode',
      GPUTextureUsage: 'globals.GPUTextureUsage',
    }),
  })],
  adapterOptions: null,
  globals: Object.freeze({ mode: 'none' }),
});

function stableJson(value) {
  return JSON.stringify(stableSortObject(value)) ?? 'null';
}

function hashStableJson(value) {
  return `sha256:${sha256Hex(stableJson(value))}`;
}

function normalizeProviders(providers) {
  if (!Array.isArray(providers) || providers.length === 0) {
    throw new Error('program bundle parity: providers must explicitly select at least one implementation.');
  }
  return providers.map((provider) => {
    if (typeof provider !== 'string' || !provider.trim()) {
      throw new Error('program bundle parity: provider entries must be non-empty strings.');
    }
    return provider.trim();
  });
}

function normalizeMode(mode) {
  if (mode !== 'contract' && mode !== 'execute') {
    throw new Error('program bundle parity: mode must be explicitly set to "contract" or "execute".');
  }
  return mode;
}

function resolveReplayPrompt(bundle) {
  const prompt = bundle.referenceTranscript?.prompt;
  if (!prompt || typeof prompt.identity !== 'string' || !prompt.identity.trim()) {
    throw new Error('program bundle parity: referenceTranscript.prompt.identity is required.');
  }
  if (prompt.identity === 'promptInput' || prompt.identity === 'metrics.promptInput') {
    throw new Error(
      'program bundle parity: bundle prompt identity is not replayable. ' +
      'Export from a report that records a concrete prompt string.'
    );
  }
  return prompt.identity;
}

function resolveModelUrl(bundle, repoRoot) {
  const manifestPath = bundle.sources?.manifest?.path;
  if (typeof manifestPath !== 'string' || !manifestPath.trim()) return null;
  return pathToFileURL(path.dirname(path.resolve(repoRoot, manifestPath))).href;
}

function summarizeReference(bundle) {
  return {
    executionGraphHash: bundle.sources.executionGraph.hash,
    tokenHash: bundle.referenceTranscript.tokens.generatedTokenIdsHash,
    textHash: bundle.referenceTranscript.output.textHash,
    tokensGenerated: bundle.referenceTranscript.output.tokensGenerated,
    stopReason: bundle.referenceTranscript.output.stopReason,
    kvCacheStateHash: bundle.referenceTranscript.kvCache.stateHash,
  };
}

function compareTranscript(bundle, transcript) {
  if (!transcript || typeof transcript !== 'object') {
    return { matched: false, reason: 'provider result did not include metrics.referenceTranscript' };
  }
  const expected = summarizeReference(bundle);
  const observed = {
    executionGraphHash: transcript.executionGraphHash ?? null,
    tokenHash: transcript.tokens?.generatedTokenIdsHash ?? null,
    textHash: transcript.output?.textHash ?? null,
    tokensGenerated: transcript.output?.tokensGenerated ?? null,
    stopReason: transcript.output?.stopReason ?? null,
    kvCacheStateHash: transcript.kvCache?.stateHash ?? null,
  };
  const mismatches = Object.keys(expected)
    .filter((key) => observed[key] !== expected[key])
    .map((key) => ({ key, expected: expected[key], observed: observed[key] }));
  return { matched: mismatches.length === 0, expected, observed, mismatches };
}

function resultBase(provider) {
  return {
    provider,
    schemaValid: true,
    providerAvailable: false,
    executed: false,
    transcriptMatched: false,
  };
}

function failureDetail(error) {
  return error instanceof Error ? error.message : String(error);
}

async function probeNodeWebGpu(options) {
  const bootstrap = await bootstrapNodeWebGPU(
    {
      ...(options.nodeWebGPUProviderOptions
        ? { providerOptions: options.nodeWebGPUProviderOptions }
        : {}),
      ...(options.nodeWebGPUContractModule
        ? { providerContractModule: options.nodeWebGPUContractModule }
        : {}),
    },
  );
  const receipt = bootstrap.receipt ?? null;
  if (!bootstrap.ok) {
    return { available: false, receipt, reason: bootstrap.detail ?? 'provider unavailable' };
  }
  await releaseNodeWebGPU();
  return { available: true, receipt, reason: null };
}

async function probeDoeWebGpu(options) {
  try {
    const contract = await import('doe-gpu/node-webgpu');
    const probe = await contract.probeNodeWebGPU(
      options.doeProviderOptions ?? DOE_NATIVE_PROVIDER_OPTIONS,
    );
    if (!probe.ok) {
      return { available: false, receipt: probe.receipt, reason: probe.error.message };
    }
    await probe.session.close();
    return { available: true, receipt: probe.receipt, reason: null };
  } catch (error) {
    return { available: false, receipt: error?.receipt ?? null, reason: failureDetail(error) };
  }
}

async function executeNodeWebGpu(bundle, options) {
  const bootstrap = await bootstrapNodeWebGPU(
    {
      ...(options.nodeWebGPUProviderOptions
        ? { providerOptions: options.nodeWebGPUProviderOptions }
        : {}),
      ...(options.nodeWebGPUContractModule
        ? { providerContractModule: options.nodeWebGPUContractModule }
        : {}),
    },
  );
  if (!bootstrap.ok) {
    return { available: false, receipt: bootstrap.receipt, reason: bootstrap.detail };
  }
  try {
    try {
      const repoRoot = path.resolve(options.repoRoot || process.cwd());
      const modelUrl = resolveModelUrl(bundle, repoRoot);
      if (!modelUrl) {
        throw new Error('program bundle parity: cannot resolve modelUrl from sources.manifest.path.');
      }
      const envelope = await runNodeCommandExecution({
        command: 'verify',
        workload: 'inference',
        modelId: bundle.modelId,
        modelUrl,
        inferenceInput: {
          prompt: resolveReplayPrompt(bundle),
          maxTokens: bundle.referenceTranscript.output.tokensGenerated,
        },
      }, options.nodeOptions ?? {});
      const comparison = compareTranscript(
        bundle,
        envelope.result?.metrics?.referenceTranscript ?? null,
      );
      return {
        available: true,
        executed: true,
        receipt: bootstrap.receipt,
        envelope,
        comparison,
      };
    } catch (error) {
      return {
        available: true,
        executed: false,
        receipt: bootstrap.receipt,
        reason: failureDetail(error),
      };
    }
  } finally {
    await releaseNodeWebGPU();
  }
}

async function executeDoeWebGpu(bundlePath, options) {
  const probe = await probeDoeWebGpu(options);
  if (!probe.available) return probe;
  try {
    if (!bundlePath) {
      throw new Error('node:doe-gpu execution requires bundlePath so packaged source bytes can be verified.');
    }
    const doe = await import('doe-gpu/program-bundle');
    const run = await doe.runProgramBundle({
      programBundlePath: bundlePath,
      providerOptions: options.doeProviderOptions ?? DOE_NATIVE_PROVIDER_OPTIONS,
      ...(options.doeExecution ? { execution: options.doeExecution } : {}),
    });
    return { available: true, executed: run.executed, run };
  } catch (error) {
    return {
      available: true,
      executed: false,
      reason: failureDetail(error),
      receipt: error?.receipt ?? probe.receipt,
    };
  }
}

async function checkProvider(provider, mode, bundle, bundlePath, options) {
  if (provider === 'browser-webgpu') {
    return {
      ...resultBase(provider),
      status: 'bundled-reference-only',
      providerAvailable: null,
      executed: null,
      transcriptMatched: null,
      referenceAvailable: true,
    };
  }

  if (provider === 'node:webgpu') {
    if (mode === 'contract') {
      const probe = await probeNodeWebGpu(options);
      return {
        ...resultBase(provider),
        status: probe.available ? 'available-unexecuted' : 'unavailable',
        providerAvailable: probe.available,
        providerReceipt: probe.receipt,
        ...(probe.reason ? { reason: probe.reason } : {}),
      };
    }
    const execution = await executeNodeWebGpu(bundle, options);
    return {
      ...resultBase(provider),
      status: !execution.available
        ? 'unavailable'
        : (!execution.executed
          ? 'available-execution-failed'
          : (execution.comparison.matched ? 'matched' : 'mismatched')),
      providerAvailable: execution.available,
      executed: execution.executed ?? false,
      transcriptMatched: execution.comparison?.matched ?? false,
      providerReceipt: execution.receipt ?? null,
      ...(execution.reason ? { reason: execution.reason } : {}),
      ...(execution.comparison ? { comparison: execution.comparison } : {}),
    };
  }

  if (provider === 'node:doe-gpu') {
    if (mode === 'contract') {
      const probe = await probeDoeWebGpu(options);
      return {
        ...resultBase(provider),
        status: probe.available ? 'available-unexecuted' : 'unavailable',
        providerAvailable: probe.available,
        providerReceipt: probe.receipt,
        ...(probe.reason ? { reason: probe.reason } : {}),
      };
    }
    const execution = await executeDoeWebGpu(bundlePath, options);
    const run = execution.run ?? null;
    return {
      ...resultBase(provider),
      status: !execution.available
        ? 'unavailable'
        : (execution.reason
          ? 'available-execution-failed'
          : (run.executed
          ? (run.transcriptMatched ? 'matched' : 'mismatched')
          : 'available-unexecuted')),
      providerAvailable: execution.available,
      executed: run?.executed ?? false,
      transcriptMatched: run?.transcriptMatched ?? false,
      providerReceipt: run?.providerReceipt ?? execution.receipt ?? null,
      ...(execution.reason ? { reason: execution.reason } : {}),
      ...(run?.transcriptComparison ? { comparison: run.transcriptComparison } : {}),
    };
  }

  throw new Error(`program bundle parity: unsupported provider "${provider}".`);
}

export async function checkProgramBundleParity(options = {}) {
  const mode = normalizeMode(options.mode);
  const providers = normalizeProviders(options.providers);
  const bundlePath = options.bundlePath ? path.resolve(options.bundlePath) : null;
  const bundle = options.bundle
    ?? (bundlePath ? (await verifyClosedProgramBundle(bundlePath)).bundle : null);
  if (!bundle) {
    throw new Error('program bundle parity: bundle or bundlePath is required.');
  }
  if (options.bundle && bundlePath) {
    await verifyClosedProgramBundle(bundlePath, options.bundle);
  } else if (options.bundle) {
    validateProgramBundle(options.bundle);
  }

  const results = [];
  for (const provider of providers) {
    results.push(await checkProvider(provider, mode, bundle, bundlePath, options));
  }

  const executionRows = results.filter((result) => result.provider !== 'browser-webgpu');
  const ok = mode === 'contract'
    ? results.every((result) => result.schemaValid === true)
    : executionRows.length > 0
      && executionRows.every((result) => result.executed === true && result.transcriptMatched === true);
  return {
    schema: PROGRAM_BUNDLE_PARITY_SCHEMA_ID,
    authority: 'portability-diagnostic-only',
    modelPromotionAuthority: false,
    ok,
    mode,
    schemaValid: true,
    bundleId: bundle.bundleId,
    modelId: bundle.modelId,
    executionGraphHash: bundle.sources.executionGraph.hash,
    reference: summarizeReference(bundle),
    tokenEvidence: buildDeterministicTokenEvidenceFromReferenceTranscript(bundle.referenceTranscript),
    providers: results,
    parityHash: hashStableJson({
      bundleId: bundle.bundleId,
      mode,
      reference: summarizeReference(bundle),
      providers: results.map((result) => ({
        provider: result.provider,
        schemaValid: result.schemaValid,
        providerAvailable: result.providerAvailable,
        executed: result.executed,
        transcriptMatched: result.transcriptMatched,
      })),
    }),
  };
}
