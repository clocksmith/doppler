import { spawn, execFileSync } from 'node:child_process';
import { writeFileSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
const inventory = () => execFileSync('ps', ['-eo', 'pid,ppid,stat,etime,args'], { encoding: 'utf8' }).split('\n')
  .filter(line => /native-participant|run-mesh-contributor|partition.*fixture|provider\.js|reference.*replay/.test(line));
const record = { startedAt: new Date().toISOString(), controllerPid: process.pid, before: inventory(), stdout: '', stderr: '', signals: [] };
record.sourceSha256 = createHash('sha256').update(readFileSync('control.js')).digest('hex');
const expected = JSON.parse(readFileSync('reservation.json'));
if (record.sourceSha256 !== expected.sources['control.js']) throw Error('Frozen provider control changed');
record.reservation = expected;
const child = spawn(process.execPath, ['control.js', '/var/tmp/reploid-ordered-native-d29cdec8-20261013/reploid', 'result.json'], { stdio: ['ignore', 'pipe', 'pipe'] });
record.pid = child.pid;
console.log(JSON.stringify({ ownedProviderPid: child.pid, rootReservation: expected.id, operationDeadlineMs: 300000, processDeadlineMs: 310000 }));
writeFileSync('process.json', JSON.stringify(record, null, 2));
let killTimer;
const timer = setTimeout(() => { record.signals.push({ signal: 'SIGTERM', at: new Date().toISOString(), reason: '310-second total process bound' }); child.kill('SIGTERM');
  killTimer = setTimeout(() => { record.signals.push({ signal: 'SIGKILL', at: new Date().toISOString(), reason: 'Owned child did not exit after termination' }); child.kill('SIGKILL'); }, 5000); }, 310000);
child.stdout.on('data', bytes => { record.stdout += bytes; }); child.stderr.on('data', bytes => { record.stderr += bytes; });
child.on('error', error => { record.spawnError = error.message; });
child.on('exit', (code, signal) => {
  clearTimeout(timer); clearTimeout(killTimer); record.exit = { code, signal, at: new Date().toISOString() }; record.after = inventory();
  record.naturalExit = !signal && record.signals.length === 0;
  writeFileSync('process.json', JSON.stringify(record, null, 2)); console.log(JSON.stringify({ exit: record.exit, naturalExit: record.naturalExit }));
  process.exitCode = code ?? 1;
});
