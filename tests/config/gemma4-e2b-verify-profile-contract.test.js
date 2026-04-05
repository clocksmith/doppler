import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const profile = JSON.parse(
  readFileSync(
    new URL('../../src/config/runtime/experiments/verify/gemma4-e2b-verify.json', import.meta.url),
    'utf8'
  )
);

assert.equal(
  profile.model,
  'gemma-4-e2b-it-q4k-ehf16-af32',
  'Gemma 4 E2B verify profile must be model-scoped.'
);

assert.equal(
  profile.runtime?.inference?.compute?.rangeAwareSelectiveWidening?.onTrigger,
  'fallback-plan',
  'Gemma 4 E2B verify profile must opt into explicit fallback-plan recovery.'
);

assert.equal(
  profile.runtime?.inference?.compute?.rangeAwareSelectiveWidening?.enabled,
  true,
  'Gemma 4 E2B verify profile must declare finiteness recovery explicitly.'
);

console.log('gemma4-e2b-verify-profile-contract.test: ok');
