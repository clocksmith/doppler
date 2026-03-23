import assert from 'node:assert/strict';

import { runNodeCommand } from '../../src/tooling/node-command-runner.js';

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
    /doppler-diagnose-missing-baseline-/
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
