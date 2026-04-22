import assert from 'node:assert/strict';
import path from 'node:path';
import {
  PROGRAM_BUNDLE_PARITY_SCHEMA_ID,
  checkProgramBundleParity,
} from '../../src/tooling/program-bundle-parity.js';

const bundlePath = path.join(
  process.cwd(),
  'examples/program-bundles/gemma-3-270m-it-q4k-ehf16-af32.program-bundle.json'
);

const result = await checkProgramBundleParity({
  bundlePath,
  providers: ['browser-webgpu', 'node:webgpu'],
});

assert.equal(result.schema, PROGRAM_BUNDLE_PARITY_SCHEMA_ID);
assert.equal(result.ok, true);
assert.equal(result.mode, 'contract');
assert.equal(result.providers.length, 2);
assert.equal(result.providers[0].provider, 'browser-webgpu');
assert.equal(result.providers[0].status, 'reference');
assert.equal(result.providers[0].comparison.ok, true);
assert.equal(result.providers[1].provider, 'node:webgpu');
assert.equal(result.providers[1].status, 'planned');
assert.equal(result.reference.tokensGenerated, 32);

await assert.rejects(
  () => checkProgramBundleParity({
    bundlePath,
    providers: ['unsupported-provider'],
  }),
  /unsupported provider/
);

console.log('program-bundle-parity.test: ok');
