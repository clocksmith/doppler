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
  mode: 'contract',
  nodeWebGPUContractModule: new URL('../fixtures/provider-v1-contract.js', import.meta.url).href,
  nodeWebGPUProviderOptions: {
    providers: [{
      id: 'fixture-webgpu',
      kind: 'module',
      module: new URL('../fixtures/webgpu-provider-v1.js', import.meta.url).href,
      gpu: { kind: 'factory', path: 'create', args: [] },
      globals: {
        GPUBufferUsage: 'globals.GPUBufferUsage',
        GPUShaderStage: 'globals.GPUShaderStage',
        GPUMapMode: 'globals.GPUMapMode',
        GPUTextureUsage: 'globals.GPUTextureUsage',
      },
    }],
    adapterOptions: null,
    globals: { mode: 'none' },
  },
});

assert.equal(result.schema, PROGRAM_BUNDLE_PARITY_SCHEMA_ID);
assert.equal(result.authority, 'portability-diagnostic-only');
assert.equal(result.modelPromotionAuthority, false);
assert.equal(result.tokenEvidence.schema, 'doppler.deterministic-token-evidence/v1');
assert.equal(result.ok, true);
assert.equal(result.schemaValid, true);
assert.equal(result.mode, 'contract');
assert.equal(result.providers.length, 2);
assert.equal(result.providers[0].provider, 'browser-webgpu');
assert.equal(result.providers[0].status, 'bundled-reference-only');
assert.equal(result.providers[0].schemaValid, true);
assert.equal(result.providers[0].providerAvailable, null);
assert.equal(result.providers[0].executed, null);
assert.equal(result.providers[0].transcriptMatched, null);
assert.equal(result.providers[1].provider, 'node:webgpu');
assert.equal(result.providers[1].status, 'available-unexecuted');
assert.equal(result.providers[1].providerAvailable, true);
assert.equal(result.providers[1].executed, false);
assert.equal(result.providers[1].transcriptMatched, false);
assert.equal(result.reference.tokensGenerated, 32);

await assert.rejects(
  () => checkProgramBundleParity({
    bundlePath,
    providers: ['unsupported-provider'],
    mode: 'contract',
  }),
  /unsupported provider/
);

await assert.rejects(
  () => checkProgramBundleParity({ bundlePath, providers: ['browser-webgpu'] }),
  /mode must be explicitly set/
);

await assert.rejects(
  () => checkProgramBundleParity({ bundlePath, mode: 'contract' }),
  /providers must explicitly select/
);

console.log('program-bundle-parity.test: ok');
