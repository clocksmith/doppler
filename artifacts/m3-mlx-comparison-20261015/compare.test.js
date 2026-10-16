import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { compare } from './compare.js';
const read = async name => JSON.parse(await fs.readFile(new URL(name, import.meta.url)));
const left = await read('mlx.json'), right = await read('doppler.json');
assert.equal(compare(left, right).exactTokenAgreement, true);
for (const mutate of [
  result => { result.complete = false; },
  result => { result.runs.pop(); },
  result => { result.policy.shared.maxTokens++; },
  result => { result.policy.source.revision = 'different'; },
  result => { result.runs[0].promptTokenIds[0]++; },
  result => { result.runs[0].evidence.tokenIds[0]++; },
  result => { result.runs[0].evidence.tokenIds.pop(); },
  result => { result.runs[0].tokenTimes[1] = NaN; },
  result => { result.runs[0].tokenTimes[1] = 0; },
]) {
  const invalid = structuredClone(right);
  mutate(invalid);
  assert.throws(() => compare(left, invalid));
}
console.log('Matched comparison rejects incomplete runs, identity drift, token mismatches, and invalid timing.');
