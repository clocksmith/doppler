import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const profilesDir = path.join(repoRoot, 'src/config/runtime/profiles');

const PROFILE_FILES = [
  'gemma4-e2b-throughput.json',
  'gemma4-e2b-low-memory.json',
];

const profiles = {};
for (const file of PROFILE_FILES) {
  const filePath = path.join(profilesDir, file);
  profiles[file] = JSON.parse(await fs.readFile(filePath, 'utf8'));
}

for (const [file, profile] of Object.entries(profiles)) {
  assert.equal(
    profile.model,
    'gemma-4-e2b-it-q4k-ehf16-af32',
    `${file}: profile must be model-scoped for Gemma 4 E2B`
  );
  assert.ok(profile.id.startsWith('profiles/'), `${file}: id must be in the profiles namespace`);
  assert.ok(profile.runtime, `${file}: must define a runtime payload`);
}

{
  const throughput = profiles['gemma4-e2b-throughput.json'];
  assert.equal(throughput.extends, 'profiles/throughput');
  assert.equal(throughput.intent, 'calibrate');
  assert.equal(throughput.runtime.inference?.batching?.batchSize, 8);
  assert.equal(throughput.runtime.inference?.batching?.readbackInterval, 8);
  assert.equal(throughput.runtime.inference?.batching?.readbackMode, 'overlapped');
  assert.equal(throughput.runtime.inference?.session?.decodeLoop?.batchSize, 8);
  assert.equal(throughput.runtime.inference?.session?.decodeLoop?.readbackInterval, 8);
  assert.equal(throughput.runtime.inference?.session?.decodeLoop?.readbackMode, 'overlapped');
  assert.equal(throughput.runtime.inference?.session?.decodeLoop?.disableCommandBatching, false);
}

{
  const lowMemory = profiles['gemma4-e2b-low-memory.json'];
  assert.equal(lowMemory.extends, 'profiles/low-memory');
  assert.equal(lowMemory.intent, 'investigate');
  assert.equal(lowMemory.runtime.shared?.bufferPool?.budget?.hardFailOnBudgetExceeded, true);
  assert.ok(lowMemory.runtime.shared?.bufferPool?.budget?.maxTotalBytes > 0);
  assert.equal(lowMemory.runtime.loading?.memoryManagement?.budget?.enabled, true);
  assert.equal(lowMemory.runtime.loading?.memoryManagement?.budget?.systemMemoryFraction, 0.5);
  assert.equal(lowMemory.runtime.loading?.memoryManagement?.budget?.maxResidentBytes, null);
  assert.equal(lowMemory.runtime.inference?.kvcache?.maxSeqLen, 2048);
  assert.equal(lowMemory.runtime.inference?.kvcache?.layout, 'contiguous');
  assert.equal(lowMemory.runtime.inference?.session?.kvcache?.maxSeqLen, 2048);
  assert.equal(lowMemory.runtime.inference?.session?.kvcache?.layout, 'contiguous');
  assert.equal(lowMemory.runtime.inference?.session?.kvcache?.tiering?.mode, 'off');
  assert.equal(lowMemory.runtime.inference?.session?.kvcache?.quantization?.mode, 'turboquant_prod');
  assert.equal(lowMemory.runtime.inference?.session?.kvcache?.quantization?.prodMode, true);
}

console.log('gemma4-e2b-runtime-profiles-contract.test: ok');
