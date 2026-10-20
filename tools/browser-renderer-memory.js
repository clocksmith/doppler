import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execute = promisify(execFile);

export function parseMacProcesses(text, profile) {
  return text.trim().split('\n').filter(Boolean).map(line => {
    const match = line.match(/^\s*(\d+)\s+(\d+)\s+(.*)$/);
    assert(match, 'Unrecognized process observation');
    return { pid: Number(match[1]), parent: Number(match[2]),
      root: match[3].includes('--user-data-dir=' + profile),
      renderer: match[3].includes('--type=renderer') };
  });
}

export async function rendererPids(profile) {
  const records = [];
  if (process.platform === 'darwin') {
    const { stdout } = await execute('ps', ['-axo', 'pid=,ppid=,command=']);
    records.push(...parseMacProcesses(stdout, profile));
  } else for (const name of await fs.readdir('/proc')) {
    if (!/^\d+$/.test(name)) continue;
    try {
      const [status, command] = await Promise.all([fs.readFile(`/proc/${name}/status`, 'utf8'), fs.readFile(`/proc/${name}/cmdline`, 'utf8')]);
      records.push({ pid: Number(name), parent: Number(status.match(/^PPid:\s+(\d+)/m)[1]),
        root: command.includes('--user-data-dir=' + profile), renderer: command.includes('--type=renderer') });
    } catch (error) { if (!['ENOENT', 'ESRCH', 'EACCES'].includes(error.code)) throw error; }
  }
  const owned = new Set(records.filter(row => row.root).map(row => row.pid));
  for (let previous = -1; previous !== owned.size;) {
    previous = owned.size;
    for (const row of records) if (owned.has(row.parent)) owned.add(row.pid);
  }
  const pids = records.filter(row => owned.has(row.pid) && row.renderer).map(row => row.pid);
  assert(pids.length, 'Physical browser renderer process must be identified.');
  return pids;
}

export async function rendererRss(pids) {
  if (process.platform === 'darwin') {
    assert(pids.length > 0 && pids.every(pid => Number.isSafeInteger(pid) && pid > 0));
    const { stdout } = await execute('ps', ['-o', 'rss=', '-p', pids.join(',')]);
    const samples = stdout.trim().split(/\s+/).map(Number);
    assert(samples.length === pids.length && samples.every(value => Number.isFinite(value) && value > 0),
      'Every owned renderer must have an RSS observation');
    return samples.reduce((sum, value) => sum + value * 1024, 0);
  }
  let bytes = 0;
  for (const pid of pids) {
    const status = await fs.readFile(`/proc/${pid}/status`, 'utf8');
    bytes += Number(status.match(/^VmRSS:\s+(\d+)/m)[1]) * 1024;
  }
  return bytes;
}
