import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { mergeKernelPathPolicy } from '../../src/config/merge-helpers.js';
import {
  buildCompareSection,
  buildDopplerRuntimeConfig,
  buildSharedBenchmarkContract,
  loadModelCatalogBundle,
  normalizeCompareLoadModeDefaults,
  parseArgs as parseCompareArgs,
  parseJsonBlock,
  parseOnOff as parseCompareOnOff,
  redactSecrets,
  usage as renderCompareUsage,
  renderComparePrompt,
  resolveCompareOwnedPromptRenderer,
  resolveCatalogTransformersjsBenchmarkTarget,
  resolveCompareProfile,
  resolveCompareLoadModes,
  resolveDopplerModelSource,
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
  const qwen08Profile = compareConfig.modelProfiles.find((entry) => entry?.dopplerModelId === 'qwen-3-5-0-8b-q4k-ehaf16') || null;
  const qwen2Profile = compareConfig.modelProfiles.find((entry) => entry?.dopplerModelId === 'qwen-3-5-2b-q4k-ehaf16') || null;
  const gemma4Profile = compareConfig.modelProfiles.find((entry) => entry?.dopplerModelId === 'gemma-4-e2b-it-q4k-ehf16-af32') || null;
  const gemma4Int4PleProfile = compareConfig.modelProfiles.find((entry) => entry?.dopplerModelId === 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple') || null;
  const gemma4Catalog = (Array.isArray(catalog.models) ? catalog.models : [])
    .find((entry) => entry?.modelId === 'gemma-4-e2b-it-q4k-ehf16-af32') || null;

  assert.deepEqual(compareConfig.defaults, {
    warmLoadMode: 'opfs',
    coldLoadMode: 'http',
  });
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
  assert.equal(qwen2Profile.compareLane, 'capability_only');
  assert.match(qwen2Profile.compareLaneReason, /correctness-clean fixture/i);
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
  assert.equal(benchmarkPolicy?.decodeProfiles?.profiles?.parity?.readbackInterval, 4);
  assert.equal(benchmarkPolicy?.decodeProfiles?.profiles?.throughput?.stopCheckMode, 'batch');
  assert.equal(benchmarkPolicy?.decodeProfiles?.profiles?.throughput?.readbackInterval, 4);
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
  assert.match(result.stdout || renderCompareUsage(), /--doppler-stop-check-mode <batch\|per-token>/);
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
      generatedText: 'Coherent TJS output.',
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
