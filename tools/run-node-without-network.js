#!/usr/bin/env node
import { spawn } from 'node:child_process';

// Qualification-only Linux wrapper. util-linux `unshare` runs Node and its
// descendants in a new user and network namespace that has no routes, so IPv4
// and IPv6 connections fail; filesystem Unix sockets and GPU device nodes still
// work. `unshare` and unprivileged user namespaces are probe prerequisites, not
// application dependencies. A self-test proves isolation before the script runs.
// No uid/gid map is written: Ubuntu 23.10+ (kernel.apparmor_restrict_unprivileged_userns=1)
// refuses an unprivileged uid_map write, but still lets the caller create the
// network namespace together with the user namespace. Inside, the script runs as
// the overflow user (nobody), while the kernel keeps checking file and device
// access against the caller's real uid and groups.
if (!process.argv[2]) throw new Error('Usage: node tools/run-node-without-network.js <node-script> [...args]');
if (process.platform !== 'linux') throw new Error('run-node-without-network requires Linux network namespaces.');

// Runs inside the namespace: prove isolation, then run the target script there.
const ISOLATED_ENTRY = `
const net = require('node:net');
const { spawn } = require('node:child_process');
const fail = (message) => { console.error('network isolation self-test failed: ' + message); process.exit(1); };
let pending = 2;
for (const host of ['192.0.2.1', '2001:db8::1']) {
  const socket = net.connect({ host, port: 9, timeout: 2000 });
  socket.once('connect', () => fail(host + ' reachable'));
  socket.once('timeout', () => fail(host + ' routable'));
  socket.once('error', (error) => {
    if (!['ENETUNREACH', 'EHOSTUNREACH', 'EADDRNOTAVAIL'].includes(error.code)) fail(host + ' ' + error.code);
    if (--pending > 0) return;
    const child = spawn(process.execPath, process.argv.slice(1), { stdio: 'inherit' });
    child.on('exit', (code, signal) => process.exit(signal ? 1 : code));
  });
}
`;

const child = spawn('unshare', ['--user', '--net', '--', process.execPath, '-e', ISOLATED_ENTRY, '--', ...process.argv.slice(2)], { stdio: 'inherit' });
child.on('error', error => { console.error(`unshare: ${error.message}`); process.exitCode = 1; });
child.on('exit', (code, signal) => { process.exitCode = signal ? 1 : code; });
