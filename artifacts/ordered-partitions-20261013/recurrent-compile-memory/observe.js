// Diagnostic supervisor only. The systemd scope enforces the independent RAM limit.
import { spawn } from 'node:child_process';
import { readFileSync, appendFileSync } from 'node:fs';
const [probe, root, shader, events, receipt] = process.argv.slice(2);
const cgroup = readFileSync('/proc/self/cgroup', 'utf8').trim().split(':').at(-1);
const append = value => appendFileSync(receipt, JSON.stringify({ at: new Date().toISOString(), ...value }) + '\n');
const memory = () => Object.fromEntries(['memory.current', 'memory.peak', 'memory.events'].map(key =>
  [key, readFileSync('/sys/fs/cgroup' + cgroup + '/' + key, 'utf8').trim()]));
const child = spawn(process.execPath, [probe, root, shader, events], { stdio: 'inherit' });
append({ phase: 'provider-started', providerPid: child.pid, supervisorPid: process.pid, cgroup });
const monitor = setInterval(() => append({ phase: 'memory', ...memory() }), 1000);
let forced;
const deadline = setTimeout(() => {
  append({ phase: 'deadline', ...memory() });
  child.kill('SIGTERM');
  forced = setTimeout(() => child.kill('SIGKILL'), 5000);
}, 290000);
child.on('exit', (code, signal) => {
  clearInterval(monitor); clearTimeout(deadline); clearTimeout(forced);
  append({ phase: 'provider-exit', code, signal, ...memory(),
    cleanup: 'Read provider events; exit alone does not establish GPU settlement.' });
  process.exitCode = code === 0 ? 0 : 1;
});
child.on('error', error => {
  clearInterval(monitor); clearTimeout(deadline); clearTimeout(forced);
  append({ phase: 'spawn-error', message: error.message }); process.exitCode = 1;
});
