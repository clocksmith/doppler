import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { loadProgramBundle } from './program-bundle.js';
import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const PROGRAM_BUNDLE_PARITY_SCHEMA_ID = 'doppler.program-bundle-parity/v1';

function stableJson(value) {
  return JSON.stringify(stableSortObject(value)) ?? 'null';
}

function hashStableJson(value) {
  return `sha256:${sha256Hex(stableJson(value))}`;
}

function normalizeProviders(providers) {
  const raw = Array.isArray(providers) && providers.length > 0
    ? providers
    : ['browser-webgpu', 'node:webgpu', 'node:doe-gpu'];
  return raw.map((provider) => {
    if (typeof provider !== 'string' || !provider.trim()) {
      throw new Error('program bundle parity: provider entries must be non-empty strings.');
    }
    return provider.trim();
  });
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
  if (typeof manifestPath !== 'string' || !manifestPath.trim()) {
    return null;
  }
  const modelDir = path.dirname(path.resolve(repoRoot, manifestPath));
  return pathToFileURL(modelDir).href;
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
    return {
      ok: false,
      reason: 'provider result did not include metrics.referenceTranscript',
    };
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
  return {
    ok: mismatches.length === 0,
    expected,
    observed,
    mismatches,
  };
}

async function checkDoeProviderAvailability() {
  try {
    const doe = await import('doe-gpu');
    return {
      ok: true,
      exports: Object.keys(doe).sort(),
    };
  } catch (error) {
    return {
      ok: false,
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

async function runNodeWebGpuProvider(bundle, options) {
  const { runNodeCommand } = await import('./node-command-runner.js');
  const repoRoot = path.resolve(options.repoRoot || process.cwd());
  const prompt = resolveReplayPrompt(bundle);
  const maxTokens = bundle.referenceTranscript.output.tokensGenerated;
  const modelUrl = resolveModelUrl(bundle, repoRoot);
  if (!modelUrl) {
    throw new Error('program bundle parity: cannot resolve modelUrl from sources.manifest.path.');
  }
  const envelope = await runNodeCommand({
    command: 'verify',
    workload: 'inference',
    modelId: bundle.modelId,
    modelUrl,
    inferenceInput: {
      prompt,
      maxTokens,
    },
  }, options.nodeOptions ?? {});
  const transcript = envelope.result?.metrics?.referenceTranscript ?? null;
  return {
    envelope,
    comparison: compareTranscript(bundle, transcript),
  };
}

export async function checkProgramBundleParity(options = {}) {
  const bundle = options.bundle ?? await loadProgramBundle(options.bundlePath);
  const providers = normalizeProviders(options.providers);
  const mode = options.mode || 'contract';
  if (mode !== 'contract' && mode !== 'execute') {
    throw new Error('program bundle parity: mode must be "contract" or "execute".');
  }
  const results = [];

  for (const provider of providers) {
    if (provider === 'browser-webgpu') {
      results.push({
        provider,
        status: 'reference',
        ok: true,
        comparison: compareTranscript(bundle, bundle.referenceTranscript),
      });
      continue;
    }

    if (provider === 'node:doe-gpu') {
      const availability = await checkDoeProviderAvailability();
      results.push({
        provider,
        status: availability.ok ? 'available-unexecuted' : 'unavailable',
        ok: mode === 'contract' ? true : false,
        availability,
      });
      continue;
    }

    if (provider === 'node:webgpu') {
      if (mode === 'contract') {
        results.push({
          provider,
          status: 'planned',
          ok: true,
          replay: {
            modelId: bundle.modelId,
            promptHash: bundle.referenceTranscript.prompt.hash,
            maxTokens: bundle.referenceTranscript.output.tokensGenerated,
          },
        });
        continue;
      }
      const run = await runNodeWebGpuProvider(bundle, options);
      results.push({
        provider,
        status: run.comparison.ok ? 'passed' : 'failed',
        ok: run.comparison.ok,
        comparison: run.comparison,
      });
      continue;
    }

    throw new Error(`program bundle parity: unsupported provider "${provider}".`);
  }

  const ok = results.every((result) => result.ok === true);
  return {
    schema: PROGRAM_BUNDLE_PARITY_SCHEMA_ID,
    ok,
    mode,
    bundleId: bundle.bundleId,
    modelId: bundle.modelId,
    executionGraphHash: bundle.sources.executionGraph.hash,
    reference: summarizeReference(bundle),
    providers: results,
    parityHash: hashStableJson({
      bundleId: bundle.bundleId,
      mode,
      reference: summarizeReference(bundle),
      providers: results.map((result) => ({
        provider: result.provider,
        status: result.status,
        ok: result.ok,
      })),
    }),
  };
}
