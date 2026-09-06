import assert from 'node:assert/strict';

import { runNodeCommand } from '../../src/tooling/node-command-runner.js';

// A synthetic provider contract isolates request routing and error propagation
// from host GPU availability and the installed optional provider's version.
const providerContractModule = 'data:text/javascript,' + encodeURIComponent(`
  export const NODE_WEBGPU_PROVIDER_SCHEMA = 'doe.webgpu-provider/v1';
  export async function openNodeWebGPU(options) {
    throw new Error('Selected diagnostic provider unavailable: ' + options.providers[0].module);
  }
`);
await assert.rejects(
  () => runNodeCommand({
    command: 'diagnose',
    modelId: 'gemma-3-1b-it-f16-af32',
    baselineProvider: 'doppler-diagnose-missing-baseline',
    observedProvider: 'doppler-diagnose-missing-observed',
  }, { providerContractModule }),
  /Selected diagnostic provider unavailable: doppler-diagnose-missing-baseline/
);
console.log('node-diagnose-command-surface.test: ok (synthetic provider contract)');

// A resolvable module can still lack the required provider API.
for (const [contract, code, stage] of [
  ['data:text/javascript,export const NODE_WEBGPU_PROVIDER_SCHEMA = "old-api";',
    'DOPPLER_PROVIDER_CONTRACT_INVALID', 'contract.validate'],
  ['doppler-diagnose-absent-contract-fixture',
    'DOPPLER_PROVIDER_CONTRACT_UNAVAILABLE', 'contract.import'],
]) {
  await assert.rejects(() => runNodeCommand({ command: 'diagnose', modelId: 'gemma-3-1b-it-f16-af32' },
    { providerContractModule: contract }), error => {
    assert.equal(error.cause?.code, code);
    assert.equal(error.cause?.stage, stage);
    return true;
  });
}
