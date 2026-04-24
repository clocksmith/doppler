import assert from 'node:assert/strict';

import { normalizeToolingCommandRequest } from '../../src/tooling/command-api.js';

const request = normalizeToolingCommandRequest({
  command: 'refresh-integrity',
  modelDir: 'models/local/test-model',
  manifestPath: 'models/local/test-model/manifest.json',
  blockSize: 1048576,
  dryRun: true,
  skipShardCheck: true,
});

assert.equal(request.command, 'refresh-integrity');
assert.equal(request.modelDir, 'models/local/test-model');
assert.equal(request.manifestPath, 'models/local/test-model/manifest.json');
assert.equal(request.blockSize, 1048576);
assert.equal(request.dryRun, true);
assert.equal(request.skipShardCheck, true);

console.log('command-api-refresh-integrity.test: ok');
