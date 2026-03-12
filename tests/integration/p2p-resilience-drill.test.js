import assert from 'node:assert/strict';
const {
  runP2PResilienceDrillCli,
} = await import('../../tools/p2p-resilience-drill.js');

const report = await runP2PResilienceDrillCli(['--stage', 'canary', '--json']);
assert.equal(report.stage, 'canary');
assert.equal(report.summary.total, 3);
assert.equal(report.summary.failed, 0);
assert.ok(Array.isArray(report.scenarios));
assert.equal(report.scenarios.length, 3);
assert.ok(report.scenarios.every((entry) => entry.status === 'pass'));

console.log('p2p-resilience-drill.test: ok');
