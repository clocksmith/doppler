import assert from 'node:assert/strict';
import { stageAnalyze, stageInspect } from '../../src/converter/forge-stages.js';

for (const modelId of ['qwen-3.8-27b', 'muse-glimmer-30b']) {
  await assert.rejects(
    stageInspect({ modelDir: `/models/${modelId}`, config: { hiddenSize: 1 } }),
    /Forge requires manifest/
  );
}

assert.throws(
  () => stageAnalyze({
    manifest: {
      architecture: {
        blockTypes: ['gated-deltanet', 'full-attention'],
        attentionGeometries: { linear: {}, full: {} },
      },
      inference: { recurrentState: {} },
    },
  }),
  /cannot represent source topology.*blockTypes.*attentionGeometries/
);

assert.throws(
  () => stageAnalyze({
    manifest: {
      architecture: { blockTypePattern: ['local', 'local', 'local', 'global'] },
      inference: {
        perceptionEncoder: { type: 'separate' },
        speculativeDrafter: { blockSize: 16 },
      },
    },
  }),
  /cannot represent source topology.*blockTypePattern.*perceptionEncoder.*speculativeDrafter/
);

console.log('✔ forge-source-truth.test.js passed');
