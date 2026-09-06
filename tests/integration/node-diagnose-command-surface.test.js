import assert from 'node:assert/strict';

import { runNodeCommand } from '../../src/tooling/node-command-runner.js';

// The provider contract is optional in a standalone checkout. Its absence must
// remain a typed import failure, not be confused with a model/provider failure.
let providerContractAvailable = true;
try {
  import.meta.resolve('doe-gpu/node-webgpu');
} catch (error) {
  assert.equal(error.code, 'ERR_MODULE_NOT_FOUND');
  providerContractAvailable = false;
}

const originalBaselineProvider = process.env.DOPPLER_DIAGNOSE_BASELINE_PROVIDER;
const originalObservedProvider = process.env.DOPPLER_DIAGNOSE_OBSERVED_PROVIDER;

try {
  process.env.DOPPLER_DIAGNOSE_BASELINE_PROVIDER = `doppler-diagnose-missing-baseline-${Date.now()}`;
  process.env.DOPPLER_DIAGNOSE_OBSERVED_PROVIDER = `doppler-diagnose-missing-observed-${Date.now()}`;

  await assert.rejects(
    () => runNodeCommand({
      command: 'diagnose',
      modelId: 'gemma-3-1b-it-f16-af32',
    }),
    (error) => {
      if (providerContractAvailable) {
        assert.match(error.message, /doppler-diagnose-missing-baseline-/);
      } else {
        assert.equal(error.cause?.code, 'DOPPLER_PROVIDER_CONTRACT_UNAVAILABLE');
        assert.equal(error.cause?.stage, 'contract.import');
        assert.match(error.message, /doe-gpu\/node-webgpu/);
      }
      return true;
    }
  );
} finally {
  if (originalBaselineProvider === undefined) {
    delete process.env.DOPPLER_DIAGNOSE_BASELINE_PROVIDER;
  } else {
    process.env.DOPPLER_DIAGNOSE_BASELINE_PROVIDER = originalBaselineProvider;
  }

  if (originalObservedProvider === undefined) {
    delete process.env.DOPPLER_DIAGNOSE_OBSERVED_PROVIDER;
  } else {
    process.env.DOPPLER_DIAGNOSE_OBSERVED_PROVIDER = originalObservedProvider;
  }
}

console.log('node-diagnose-command-surface.test: ok');
