import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

const configPath = path.join(
  repoRoot,
  'src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json'
);
const config = JSON.parse(await fs.readFile(configPath, 'utf8'));

assert.equal(config.output?.baseDir, 'models/local');
assert.equal(config.output?.modelBaseId, 'gemma-4-e2b-it-q4k-ehf16-af32');
assert.equal(config.output?.textOnly, false);

assert.equal(config.quantization?.weights, 'q4k');
assert.equal(config.quantization?.embeddings, 'f16');
assert.equal(config.quantization?.lmHead, 'f16');
assert.equal(config.quantization?.computePrecision, 'f32');
assert.equal(config.quantization?.q4kLayout, 'row');

assert.equal(config.inference?.attention?.queryPreAttnScalar, 1);
assert.equal(config.inference?.attention?.slidingWindow, 512);
assert.equal(config.inference?.attention?.valueNorm, true);
assert.equal(config.inference?.normalization?.rmsNormWeightOffset, false);
assert.equal(config.inference?.ffn?.useDoubleWideMlp, true);
assert.equal(config.inference?.chatTemplate?.type, 'gemma4');
assert.equal(config.inference?.rope?.partialRotaryFactor, 0.25);
assert.equal(config.inference?.rope?.ropeLocalPartialRotaryFactor, null);
assert.equal(config.inference?.rope?.ropeInterleaved, false);
assert.equal(config.inference?.rope?.ropeFrequencyBaseDim, 512);
assert.equal(config.inference?.rope?.ropeLocalFrequencyBaseDim, null);
assert.equal(config.inference?.rope?.ropeScalingType, null);
assert.equal(config.inference?.output?.finalLogitSoftcapping, 30);
assert.equal(config.inference?.output?.embeddingPostprocessor, null);

const expanded = expandExecutionV1(config.execution);
assert.ok(expanded.length > 0, 'execution must expand to at least one step');

assert.equal(config.session?.compute?.defaults?.activationDtype, 'f32');
assert.equal(config.session?.kvcache?.kvDtype, 'f32');

console.log('gemma4-e2b-conversion-config-contract.test: ok');
