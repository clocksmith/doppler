import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

export async function rendererPids(profile) {
  const records = [];
  for (const name of await fs.readdir('/proc')) {
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
  let bytes = 0;
  for (const pid of pids) {
    const status = await fs.readFile(`/proc/${pid}/status`, 'utf8');
    bytes += Number(status.match(/^VmRSS:\s+(\d+)/m)[1]) * 1024;
  }
  return bytes;
}
