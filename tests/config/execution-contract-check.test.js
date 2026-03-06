import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  extractExecutionContractFacts,
  validateExecutionContractFacts,
  validateManifestExecutionContract,
} from '../../src/config/execution-contract-check.js';
import { validateManifest } from '../../src/formats/rdrr/validation.js';

const translateGemmaManifest = JSON.parse(
  fs.readFileSync('models/local/translategemma-4b-it-wq4k-ef16-hf16/manifest.json', 'utf8')
);

{
  const result = validateManifestExecutionContract(translateGemmaManifest);
  assert.equal(result.ok, true);
  assert.deepEqual(
    result.checks.map((entry) => entry.ok),
    [true, true]
  );
}

{
  const conflictingManifest = structuredClone(translateGemmaManifest);
  conflictingManifest.inference.sessionDefaults.kvcache.layout = 'bdpa';
  conflictingManifest.inference.sessionDefaults.decodeLoop.batchSize = 16;
  delete conflictingManifest.inference.sessionDefaults.decodeLoop.disableCommandBatching;

  const facts = extractExecutionContractFacts(conflictingManifest);
  assert.equal(facts.session.layout, 'bdpa');
  assert.equal(facts.session.decodeBatchSize, 16);

  const executionContract = validateExecutionContractFacts(facts);
  assert.equal(executionContract.ok, false);
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('decode-only') && message.includes('prefill_attention')
    )
  );
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('disableCommandBatching=true')
    )
  );
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('batchSize <= 1')
    )
  );
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('maxSeqLen <= 2048')
    )
  );

  const validation = validateManifest(conflictingManifest);
  assert.equal(validation.valid, false);
  assert.ok(
    validation.errors.some((message) =>
      message.includes('decode-only') && message.includes('prefill_attention')
    )
  );
}

console.log('execution-contract-check.test: ok');
