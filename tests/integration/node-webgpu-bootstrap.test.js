import assert from 'node:assert/strict';
import {
  bootstrapNodeWebGPU,
  bootstrapNodeWebGPUProvider,
  releaseNodeWebGPU,
} from '../../src/tooling/node-webgpu.js';

const contractModule = new URL('../fixtures/provider-v1-contract.js', import.meta.url).href;
const contract = await import(contractModule);
const originalModule = process.env.DOPPLER_NODE_WEBGPU_MODULE;
const originalArgs = process.env.DOPPLER_NODE_WEBGPU_PROVIDER_ARGS;

try {
  delete process.env.DOPPLER_NODE_WEBGPU_MODULE;
  delete process.env.DOPPLER_NODE_WEBGPU_PROVIDER_ARGS;

  const defaultResult = await bootstrapNodeWebGPU({ providerContractModule: contractModule });
  assert.equal(defaultResult.ok, true);
  assert.equal(defaultResult.provider, 'pre-installed');
  assert.equal(defaultResult.receipt.contract, 'doe.webgpu-provider/v1');
  assert.deepEqual(contract.getLastProviderOptions().providers.map((provider) => provider.id), [
    'pre-installed',
    'webgpu',
  ]);
  assert.equal(contract.getLastProviderOptions().adapterOptions, null);

  const reused = await bootstrapNodeWebGPU({ providerContractModule: 'not-imported-while-active' });
  assert.equal(reused.session, defaultResult.session);

  const releasedDefault = await releaseNodeWebGPU();
  assert.equal(releasedDefault.released, true);
  assert.equal(releasedDefault.receipt.globals.restored, true);

  const explicit = await bootstrapNodeWebGPUProvider('custom-webgpu', {
    providerContractModule: contractModule,
    id: 'custom-provider',
    createArgs: ['backend=vulkan'],
    adapterOptions: { powerPreference: 'low-power' },
    globalMode: 'install-missing',
  });
  assert.equal(explicit.ok, true);
  assert.equal(explicit.provider, 'custom-provider');
  assert.deepEqual(contract.getLastProviderOptions(), {
    providers: [{
      id: 'custom-provider',
      kind: 'module',
      module: 'custom-webgpu',
      gpu: { kind: 'factory', path: 'create', args: [['backend=vulkan']] },
      globals: {
        GPUBufferUsage: 'globals.GPUBufferUsage',
        GPUShaderStage: 'globals.GPUShaderStage',
        GPUMapMode: 'globals.GPUMapMode',
        GPUTextureUsage: 'globals.GPUTextureUsage',
      },
    }],
    adapterOptions: { powerPreference: 'low-power' },
    globals: { mode: 'install-missing' },
  });
  await releaseNodeWebGPU();

  process.env.DOPPLER_NODE_WEBGPU_MODULE = 'environment-webgpu';
  process.env.DOPPLER_NODE_WEBGPU_PROVIDER_ARGS = 'backend=metal,adapter=discrete';
  const environmentResult = await bootstrapNodeWebGPU({ providerContractModule: contractModule });
  assert.equal(environmentResult.ok, true);
  assert.deepEqual(contract.getLastProviderOptions().providers, [{
    id: 'environment-webgpu',
    kind: 'module',
    module: 'environment-webgpu',
    gpu: {
      kind: 'factory',
      path: 'create',
      args: [['backend=metal', 'adapter=discrete']],
    },
    globals: {
      GPUBufferUsage: 'globals.GPUBufferUsage',
      GPUShaderStage: 'globals.GPUShaderStage',
      GPUMapMode: 'globals.GPUMapMode',
      GPUTextureUsage: 'globals.GPUTextureUsage',
    },
  }]);
  await releaseNodeWebGPU();

  const unavailable = await bootstrapNodeWebGPU({
    providerContractModule: `missing-provider-contract-${Date.now()}`,
  });
  assert.equal(unavailable.ok, false);
  assert.equal(unavailable.error.code, 'DOPPLER_PROVIDER_CONTRACT_UNAVAILABLE');
  assert.match(unavailable.detail, /provider-v1 contract/);

  assert.deepEqual(await releaseNodeWebGPU(), {
    released: false,
    provider: null,
    reason: 'not-owned',
    receipt: null,
  });
} finally {
  await releaseNodeWebGPU();
  if (originalModule === undefined) delete process.env.DOPPLER_NODE_WEBGPU_MODULE;
  else process.env.DOPPLER_NODE_WEBGPU_MODULE = originalModule;
  if (originalArgs === undefined) delete process.env.DOPPLER_NODE_WEBGPU_PROVIDER_ARGS;
  else process.env.DOPPLER_NODE_WEBGPU_PROVIDER_ARGS = originalArgs;
}

console.log('node-webgpu-bootstrap.test: ok');
