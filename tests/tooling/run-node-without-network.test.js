import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const WRAPPER = 'tools/run-node-without-network.js';

function run(args) {
  return spawnSync(process.execPath, [WRAPPER, ...args], { encoding: 'utf8', timeout: 30_000 });
}

if (process.platform !== 'linux') {
  const result = run(['missing.js']);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /requires Linux network namespaces/);
  console.log('run-node-without-network.test: ok (non-Linux refusal only)');
  process.exit(0);
}

const probe = spawnSync('unshare', ['--user', '--net', '--', 'true'], { encoding: 'utf8' });
if (probe.error || probe.status !== 0) {
  console.log(`run-node-without-network.test: skipped (unprivileged network namespaces unavailable: ${
    (probe.error?.message ?? probe.stderr).trim()})`);
  process.exit(0);
}

const directory = mkdtempSync(join(tmpdir(), 'doppler-no-network-'));
try {
  const script = join(directory, 'probe.js');
  writeFileSync(script, `
const net = require('node:net');
const fs = require('node:fs');
const socketPath = ${JSON.stringify(join(directory, 'local.sock'))};
const server = net.createServer((connection) => connection.end('pong')).listen(socketPath, () => {
  net.connect(socketPath).on('data', (data) => {
    server.close();
    const outbound = net.connect({ host: '1.1.1.1', port: 443, timeout: 3000 });
    outbound.once('connect', () => { console.log('outbound connected'); process.exit(2); });
    outbound.once('timeout', () => { console.log('outbound timeout'); process.exit(2); });
    outbound.once('error', (error) => {
      console.log(JSON.stringify({ unix: String(data), outbound: error.code, args: process.argv.slice(2) }));
      process.exit(7);
    });
  });
});
`);
  const result = run([script, 'first', 'second']);
  assert.equal(result.status, 7, `wrapper must pass the script's exit status through: ${result.stderr}`);
  const report = JSON.parse(result.stdout.trim().split('\n').pop());
  assert.equal(report.unix, 'pong', 'filesystem Unix sockets keep working');
  assert.ok(['ENETUNREACH', 'EHOSTUNREACH', 'EADDRNOTAVAIL'].includes(report.outbound), report.outbound);
  assert.deepEqual(report.args, ['first', 'second']);
} finally {
  rmSync(directory, { recursive: true, force: true });
}

console.log('run-node-without-network.test: ok');
