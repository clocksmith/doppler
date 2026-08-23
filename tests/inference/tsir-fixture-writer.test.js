import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { maybeWriteFixtureSnapshot } from '../../src/inference/pipelines/text/tsir-fixture-writer.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-tsir-fixture-'));

try {
  const fixture = {
    dir: root,
    layerFilter: [0],
    prefillOnly: false,
    generationStep: 2,
  };
  const options = {
    tsirFixture: fixture,
    layerIdx: 0,
    hiddenSize: 2,
  };

  const prefill = await maybeWriteFixtureSnapshot(
    'layer_in',
    new Float32Array([1, 2, 3, 4]),
    { ...options, numTokens: 2 },
  );
  assert.equal(prefill.phase, 'prefill');
  assert.equal(prefill.generationStep, 0);
  assert.equal(prefill.filePath, path.join(root, 'layer_0', 'pre_layer_input.npy'));

  fixture.currentGenerationStep = 1;
  const firstDecode = await maybeWriteFixtureSnapshot(
    'layer_in',
    new Float32Array([5, 6]),
    { ...options, numTokens: 1 },
  );
  assert.equal(firstDecode, null);

  fixture.currentGenerationStep = 2;
  const selectedDecode = await maybeWriteFixtureSnapshot(
    'layer_in',
    new Float32Array([7, 8]),
    { ...options, numTokens: 1 },
  );
  assert.equal(selectedDecode.phase, 'decode');
  assert.equal(selectedDecode.generationStep, 2);
  assert.equal(
    selectedDecode.filePath,
    path.join(root, 'generation_step_2', 'layer_0', 'pre_layer_input.npy'),
  );
  await fs.access(prefill.filePath);
  await fs.access(selectedDecode.filePath);
} finally {
  await fs.rm(root, { recursive: true, force: true });
}

console.log('tsir-fixture-writer.test: ok');
