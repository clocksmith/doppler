import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createManifestConversionPreflightReceipt } from '../../src/converter/manifest-conversion-preflight.js';

async function readJson(file) {
  return JSON.parse(await fs.readFile(file, 'utf8'));
}

const policy = await readJson('src/config/forge/conversion-preflight/glimmer-30b-text.json');
const [rawConfig, conversionConfig, semanticReceipt, headers, tensorPolicy, tensorClosureReceipt, checkedIn] = await Promise.all([
  readJson(policy.rawConfig),
  readJson(policy.conversionConfig),
  readJson(policy.semanticReceipt),
  readJson(policy.headerEvidence),
  readJson(policy.tensorPolicy),
  readJson(policy.tensorClosureReceipt),
  readJson(policy.output),
]);
const inputs = { rawConfig, conversionConfig, semanticReceipt, headers, tensorPolicy, tensorClosureReceipt, policy };
const receipt = createManifestConversionPreflightReceipt(inputs);
assert.deepEqual(receipt, checkedIn, 'conversion preflight receipt must be deterministic');
assert.equal(receipt.sourceEvidence.scopedTensorCount, 627);
assert.equal(receipt.sourceEvidence.sourceQuantization, 'bf16');
assert.deepEqual(receipt.sourceEvidence.tensorRoles, { embedding: 1, lm_head: 1, matmul: 416, norm: 209 });
assert.equal(receipt.conversionPlan.manifestQuantization, 'F16');
assert.equal(receipt.conversionPlan.executionVersion, 'v1');
assert.equal(receipt.conversionPlan.commandCount > 0, true);
assert.equal(receipt.dispositions.headerPreflightPassed, true);
assert.equal(receipt.dispositions.weightBodiesPresent, false);
assert.equal(receipt.dispositions.conversionExecuted, false);
assert.equal(receipt.dispositions.packEligible, false);

const driftedConfig = structuredClone(conversionConfig);
driftedConfig.inference.attention.queryScale = 4;
assert.throws(
  () => createManifestConversionPreflightReceipt({ ...inputs, conversionConfig: driftedConfig }),
  /Conversion config does not match/
);

const mixedHeaders = structuredClone(headers);
mixedHeaders.tensors['model.language_model.layers.0.mlp.up_proj.weight'].dtype = 'F16';
assert.throws(
  () => createManifestConversionPreflightReceipt({ ...inputs, headers: mixedHeaders }),
  /dtype mismatch/,
  'source dtype changes must invalidate tensor closure before planning'
);

const incompleteClosure = { ...tensorClosureReceipt, complete: false };
assert.throws(
  () => createManifestConversionPreflightReceipt({ ...inputs, tensorClosureReceipt: incompleteClosure }),
  /does not match its promoted evidence/,
  'unpromoted closure edits must fail before conversion'
);

const moduleSource = await fs.readFile('src/converter/manifest-conversion-preflight.js', 'utf8');
assert.doesNotMatch(moduleSource, /glimmer|qwen/i, 'generic conversion preflight must not contain model-family names');

console.log('✔ manifest-conversion-preflight.test.js passed');
