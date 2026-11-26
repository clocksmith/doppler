// GPU tier of tests/integration/emitted-enum-values.test.js: decode stats are produced only by a
// real generation, so this file loads a local model on a WebGPU device and checks that every
// value in pipeline.stats is kebab-case (decodeMode, batchGuardReason, stopReason and the rest)
// on the batched path, a forced single-token path and the lookup-speculation path.
// Skipped unless MODEL_DIR names a local RDRR model directory. Opens a GPU device.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { probeNodeGPU } from '../../tests/helpers/gpu-probe.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

const SNAKE = /^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$/;

function snakeValues(value, trail = 'stats', out = []) {
  if (typeof value === 'string') {
    if (SNAKE.test(value)) out.push(`${trail}=${value}`);
  } else if (Array.isArray(value)) {
    value.forEach((entry, index) => snakeValues(entry, `${trail}[${index}]`, out));
  } else if (value && typeof value === 'object') {
    for (const [key, entry] of Object.entries(value)) snakeValues(entry, `${trail}.${key}`, out);
  }
  return out;
}

if (!process.env.MODEL_DIR) {
  console.log('emitted-decode-stats-kebab-physical: skipped; set MODEL_DIR to a local verified model');
} else {
  const directory = process.env.MODEL_DIR;
  const manifest = JSON.parse(await fs.readFile(path.join(directory, 'manifest.json'), 'utf8'));
  assert.equal((await probeNodeGPU({ installFileFetchShim: true })).ready, true);
  const pipeline = await createPipeline(manifest, {
    storage: createNodeFileArtifactStorageContext(pathToFileURL(directory).href, manifest),
    runtimeConfig: { inference: { session: { kvcache: { maxSeqLen: 256 } } } },
  });
  try {
    const base = { maxTokens: 8, temperature: 0, topK: 1, topP: 1, useChatTemplate: true };
    const variants = [
      ['default', {}],
      ['token-constraint', { logitMaskFn: () => null }],
      ['lookup-speculation', { speculation: { mode: 'lookup', tokens: 3, maxMatchTokens: 4, minMatchTokens: 3, maxRejectedBlocks: 1 } }],
    ];
    const seen = {};
    for (const [label, extra] of variants) {
      await pipeline.reset();
      for await (const _token of pipeline.generateTokens([{ role: 'user', content: 'Name a color.' }], { ...base, ...extra })) {
        // drain
      }
      const stats = structuredClone(pipeline.stats);
      seen[label] = { decodeMode: stats.decodeMode, batchGuardReason: stats.batchGuardReason ?? null, stopReason: stats.stopReason ?? null };
      assert.deepEqual(snakeValues(stats), [], `${label}: decode stats emitted snake_case values`);
    }
    console.log('EMITTED_DECODE_STATS=' + JSON.stringify(seen));
  } finally {
    await pipeline.unload?.();
    destroyDevice();
    await releaseNodeWebGPU?.();
  }
  console.log('emitted-decode-stats-kebab-physical: ok');
}
