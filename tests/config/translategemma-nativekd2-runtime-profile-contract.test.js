import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const PROFILE_PATH = new URL(
  '../../src/config/runtime/profiles/translategemma-1b-nativekd2-q4k-rdna3-throughput-probe.json',
  import.meta.url
);

const profile = JSON.parse(await readFile(PROFILE_PATH, 'utf8'));
const session = profile.runtime?.inference?.session;

assert.equal(
  profile.id,
  'profiles/translategemma-1b-nativekd2-q4k-rdna3-throughput-probe'
);
assert.equal(profile.intent, 'calibrate');
assert.equal(profile.stability, 'experimental');
assert.equal(profile.extends, 'profiles/throughput');
assert.equal(profile.model, 'translategemma-4b-1b-enes-q4k-ehf16-af32');
assert.equal(session?.retainQ4KMaterialization, true);
assert.equal(session?.useWideTileQ4KPrefill, true);
assert.equal(session?.useSandwichRMSNormPairFusion, true);
assert.equal(session?.usePostFfnNextInputRMSNormPairFusion, true);
assert.equal(session?.useFusedQKVSplitQKNormRoPE, true);
assert.equal(profile.runtime?.inference?.batching, undefined);
assert.equal(session?.decodeLoop, undefined);

console.log('translategemma-nativekd2-runtime-profile-contract.test: ok');
