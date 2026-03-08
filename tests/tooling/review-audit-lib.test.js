import assert from 'node:assert/strict';

import {
  assignOwner,
  getAuditPaths,
  getScopeConfig,
  validateEventSequence,
  walkScope,
} from '../../tools/review-audit-lib.js';

{
  const srcScope = getScopeConfig('src');
  assert.equal(srcScope.name, 'src');
  assert.match(srcScope.auditDir, /reports\/review\/src-audit$/);
}

{
  const toolsScope = getScopeConfig('tools');
  assert.equal(toolsScope.name, 'tools');
  assert.match(toolsScope.auditDir, /reports\/review\/tools-audit$/);
}

assert.deepEqual(
  assignOwner('src', 'src/config/runtime.js'),
  { owner: 'A', agent: 'config-owner' }
);
assert.deepEqual(
  assignOwner('src', 'src/gpu/device.js'),
  { owner: 'B', agent: 'runtime-owner' }
);
assert.deepEqual(
  assignOwner('src', 'src/tooling/command-api.js'),
  { owner: 'C', agent: 'surface-owner' }
);

assert.deepEqual(
  assignOwner('tools', 'tools/configs/runtime-overlays/trace-config.json'),
  { owner: 'A', agent: 'config-owner' }
);
assert.deepEqual(
  assignOwner('tools', 'tools/vendor-bench.js'),
  { owner: 'B', agent: 'runtime-owner' }
);
assert.deepEqual(
  assignOwner('tools', 'tools/doppler-cli.js'),
  { owner: 'C', agent: 'surface-owner' }
);

{
  const paths = getAuditPaths('tools');
  assert.match(paths.eventsFile, /reports\/review\/tools-audit\/events\.jsonl$/);
  assert.match(paths.latestFile, /reports\/review\/tools-audit\/latest\.jsonl$/);
}

{
  const srcFiles = await walkScope('src');
  assert.ok(srcFiles.some(file => file.endsWith('/src/config/runtime.js')));
  assert.ok(srcFiles.every(file => file.endsWith('.js') || file.endsWith('.wgsl')));
  assert.ok(srcFiles.every(file => !file.endsWith('.d.js')));
}

{
  const toolFiles = await walkScope('tools');
  assert.ok(toolFiles.some(file => file.endsWith('/tools/doppler-cli.js')));
  assert.ok(toolFiles.some(file => file.endsWith('/tools/configs/conversion/README.md')));
  assert.ok(toolFiles.some(file => file.endsWith('/tools/policies/agent-parity-policy.json')));
}

assert.doesNotThrow(() => validateEventSequence([
  { seq: 1 },
  { seq: 2 },
  { seq: 3 },
]));
assert.throws(() => validateEventSequence([
  { seq: 1 },
  { seq: 1 },
]), /duplicate seq=1/);

console.log('review-audit-lib.test: ok');
