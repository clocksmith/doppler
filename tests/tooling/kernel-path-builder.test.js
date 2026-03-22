import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

import { getKernelPath } from '../../src/config/kernel-path-loader.js';
import { resolveMaterializedManifestFromConversionConfig } from '../../src/tooling/conversion-config-materializer.js';
import {
  buildKernelPathBuilderIndex,
  buildKernelPathBuilderProposals,
  buildKernelPathBuilderRuntimeOverlay,
} from '../../src/tooling/kernel-path-builder/index.js';

function cloneJson(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function buildManifestEntry(configPath, rawConfig) {
  const modelId = String(rawConfig?.output?.modelBaseId ?? '').trim();
  const materialized = resolveMaterializedManifestFromConversionConfig(rawConfig, {
    modelId,
    modelType: rawConfig?.modelType ?? null,
  });
  return {
    manifestPath: `models/local/${modelId}/manifest.json`,
    manifest: {
      modelId,
      modelType: materialized.modelType,
      architecture: materialized.architecture ?? null,
      inference: materialized.inference,
      metadata: {
        sourceRuntime: {
          mode: 'artifact',
        },
      },
    },
  };
}

const registryPayload = await readJson(new URL('../../src/config/kernel-paths/registry.json', import.meta.url));
const registryEntries = registryPayload.entries.map((entry) => ({
  ...entry,
  path: getKernelPath(entry.id),
}));

const gemmaConfig = await readJson(new URL('../../src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json', import.meta.url));
const qwen08Config = await readJson(new URL('../../src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json', import.meta.url));
const qwen2bConfig = await readJson(new URL('../../src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json', import.meta.url));
const proposalOnlyGemmaConfig = cloneJson(gemmaConfig);
proposalOnlyGemmaConfig.output.modelBaseId = 'gemma-3-270m-it-q4k-ehf16-af32-inline-proposal';
proposalOnlyGemmaConfig.execution.kernels.sample.constants = {
  GREEDY_ONLY: true,
};

const payload = buildKernelPathBuilderIndex({
  configEntries: [
    {
      configPath: 'src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json',
      rawConfig: gemmaConfig,
    },
    {
      configPath: 'src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json',
      rawConfig: qwen08Config,
    },
    {
      configPath: 'src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json',
      rawConfig: qwen2bConfig,
    },
    {
      configPath: 'src/config/conversion/unit/gemma-3-270m-it-q4k-ehf16-af32-inline-proposal.json',
      rawConfig: proposalOnlyGemmaConfig,
    },
  ],
  manifestEntries: [
    buildManifestEntry(
      'src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json',
      gemmaConfig
    ),
  ],
  registryEntries,
});

assert.equal(payload.models.length, 4);
assert.equal(payload.skipped.length, 0);
assert.equal(payload.stats.manifestSources, 1);
assert.equal(payload.stats.modelsWithArtifactManifest, 1);
assert.equal(payload.stats.proposalOnlyModels, 1);

const gemma = payload.models.find((entry) => entry.modelId === 'gemma-3-270m-it-q4k-ehf16-af32');
assert.ok(gemma, 'expected Gemma 270M record');
assert.equal(gemma.sources.length, 2);
assert.equal(gemma.sourceConsistency?.ok, true);
assert.equal(gemma.runtime.actualLowering, 'inline-kernel-path');
assert.ok(gemma.runtime.exactKernelPathIds.includes('gemma3-q4k-dequant-f32a-small-attn'));
assert.ok(gemma.runtime.kernelPathIds.includes('gemma3-q4k-dequant-f32a-small-attn'));
assert.ok(Array.isArray(gemma.execution.sections.decode) && gemma.execution.sections.decode.length > 0);
assert.ok(typeof gemma.execution.signature === 'string' && gemma.execution.signature.startsWith('execv1:'));
assert.ok(Array.isArray(gemma.candidate?.proposal?.path?.decode?.steps));
assert.ok(Array.isArray(gemma.candidate?.proposal?.path?.prefill?.steps));
assert.equal(gemma.candidate?.proposal?.verification?.ok, true);
assert.equal(
  gemma.candidate?.proposal?.verification?.checks?.some((entry) => entry?.id === 'roundTripShape' && entry?.ok === true),
  true
);
assert.equal(
  gemma.candidate?.proposal?.verification?.checks?.some((entry) => entry?.id === 'executionPlanCompile' && entry?.ok === true),
  true
);

const qwen = payload.models.find((entry) => entry.modelId === 'qwen-3-5-0-8b-q4k-ehaf16');
assert.ok(qwen, 'expected Qwen 0.8B record');
assert.equal(qwen.runtime.actualLowering, 'execution-graph-only');
assert.equal(qwen.candidate.available, false);
assert.match(qwen.candidate.error, /activationDtype="f16"/);
assert.equal(qwen.candidate.closestMatches[0]?.id, 'qwen3-q4k-dequant-f32a-online');
assert.ok(qwen.customRuntimeFacts.some((entry) => entry.id.endsWith('.linear_attention_runtime')));

const qwen2b = payload.models.find((entry) => entry.modelId === 'qwen-3-5-2b-q4k-ehaf16');
assert.ok(qwen2b, 'expected Qwen 2B record');
assert.equal(qwen2b.runtime.actualLowering, 'execution-graph-only');
assert.equal(qwen2b.candidate.available, false);
assert.equal(qwen2b.runtime.exactKernelPathIds.length, 0);
assert.equal(qwen2b.candidate.closestMatches[0]?.id, 'qwen3-q4k-dequant-f16a-online');
assert.equal(
  qwen2b.candidate.closestMatches[0]?.mismatchDetails?.some((entry) => entry?.code === 'custom_runtime_bypass'),
  true
);

const proposalOnly = payload.models.find((entry) => entry.modelId === 'gemma-3-270m-it-q4k-ehf16-af32-inline-proposal');
assert.ok(proposalOnly, 'expected proposal-only Gemma record');
assert.equal(proposalOnly.runtime.actualLowering, 'inline-kernel-path');
assert.equal(proposalOnly.runtime.exactKernelPathIds.length, 0);
assert.equal(proposalOnly.candidate.proposal?.kind, 'proposed');
assert.equal(proposalOnly.candidate.proposal?.verification?.ok, true);
assert.equal(proposalOnly.runtime.kernelPathIds[0], proposalOnly.candidate.proposal?.path?.id);
assert.deepEqual(
  payload.reverseIndexes.kernelPaths[proposalOnly.candidate.proposal?.path?.id],
  ['gemma-3-270m-it-q4k-ehf16-af32-inline-proposal']
);

const proposalsPayload = buildKernelPathBuilderProposals(payload);
assert.ok(proposalsPayload.stats.proposals >= 2);
assert.ok(proposalsPayload.stats.newKernelPaths >= 1);
assert.equal(
  proposalsPayload.proposals.some((entry) => entry.modelId === 'gemma-3-270m-it-q4k-ehf16-af32-inline-proposal'),
  true
);

const timedStep = gemma.execution.sections.decode[0];
const overlay = buildKernelPathBuilderRuntimeOverlay(gemma, {
  modelId: gemma.modelId,
  timestamp: '2026-03-21T12:00:00.000Z',
  runtimeProfile: 'unit',
  metrics: {
    modelLoadMs: 42,
    firstTokenMs: 55,
    decodeTokensPerSec: 123.45,
    kernelPathId: gemma.runtime.kernelPathIds[0],
    kernelPathSource: 'runtime',
    executionPlan: {
      primary: { id: 'primary' },
      finalActivePlanId: 'primary',
      transitions: [
        { kind: 'activate', to: 'primary' },
      ],
    },
    decodeProfileSteps: [
      {
        timings: {
          [timedStep.id]: 3.5,
          unmatched_timer: 0.75,
        },
      },
    ],
  },
  memory: {
    used: 8192,
    kvCache: {
      layout: 'contiguous',
    },
  },
});

assert.ok(overlay, 'expected runtime overlay');
assert.equal(overlay.kernelPathId, gemma.runtime.kernelPathIds[0]);
assert.equal(overlay.executionPlan.finalActivePlanId, 'primary');
assert.equal(overlay.stepTimingsById[timedStep.id].totalMs, 3.5);
assert.equal(overlay.unmatchedTimingLabels[0].label, 'unmatched_timer');
assert.equal(overlay.topDecodeTimers[0].label, timedStep.id);

assert.deepEqual(payload.reverseIndexes.executionGraphs[gemma.execution.signature], [
  'gemma-3-270m-it-q4k-ehf16-af32',
]);

console.log('kernel-path-builder.test: ok');
