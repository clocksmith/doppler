import assert from 'node:assert/strict';

import {
  TOOLING_COMMANDS,
  ensureCommandSupportedOnSurface,
  normalizeToolingCommandRequest,
} from '../../src/tooling/command-api.js';

assert.equal(TOOLING_COMMANDS.includes('lora'), true);
assert.equal(TOOLING_COMMANDS.includes('distill'), true);

const distillRequest = normalizeToolingCommandRequest({
  command: 'distill',
  action: 'run',
  workloadPath: 'src/training/workload-packs/distill-translategemma-tiny.json',
  subsetManifest: 'reports/training/distill/example/subset_manifest.json',
});

assert.equal(distillRequest.command, 'distill');
assert.equal(distillRequest.action, 'run');
assert.equal(
  distillRequest.workloadPath,
  'src/training/workload-packs/distill-translategemma-tiny.json'
);
assert.equal(
  distillRequest.subsetManifest,
  'reports/training/distill/example/subset_manifest.json'
);

const loraRequest = normalizeToolingCommandRequest({
  command: 'lora',
  action: 'export',
  runRoot: 'reports/training/lora/lora-toy-tiny/2026-03-07T00-00-00.000Z',
});

assert.equal(loraRequest.command, 'lora');
assert.equal(loraRequest.action, 'export');
assert.equal(
  loraRequest.runRoot,
  'reports/training/lora/lora-toy-tiny/2026-03-07T00-00-00.000Z'
);

assert.throws(
  () => ensureCommandSupportedOnSurface(distillRequest, 'browser'),
  /Node-only/
);

console.log('training-operator-command-api.test: ok');
