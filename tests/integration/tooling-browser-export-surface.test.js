import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

function runInline(args, source) {
  return spawnSync(process.execPath, [...args, '--input-type=module', '--eval', source], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
}

{
  const result = runInline([], `
    const mod = await import('@simulatte/doppler/tooling');
    if (typeof mod.runNodeCommand !== 'function') {
      throw new Error('expected runNodeCommand on default tooling import');
    }
    if (typeof mod.runBrowserCommandInNode !== 'function') {
      throw new Error('expected runBrowserCommandInNode on default tooling import');
    }
    if (typeof mod.normalizeToolingCommandRequest !== 'function') {
      throw new Error('expected normalizeToolingCommandRequest on default tooling import');
    }
  `);
  assert.equal(result.status, 0, result.stderr);
}

{
  const result = runInline(['--conditions=browser'], `
    const mod = await import('@simulatte/doppler/tooling');
    if (typeof mod.normalizeToolingCommandRequest !== 'function') {
      throw new Error('expected normalizeToolingCommandRequest on browser tooling import');
    }
    if (typeof mod.runBrowserCommand !== 'function') {
      throw new Error('expected runBrowserCommand on browser tooling import');
    }
    if ('runNodeCommand' in mod) {
      throw new Error('browser tooling import unexpectedly exposed runNodeCommand');
    }
    if ('runBrowserCommandInNode' in mod) {
      throw new Error('browser tooling import unexpectedly exposed runBrowserCommandInNode');
    }
  `);
  assert.equal(result.status, 0, result.stderr);
}

console.log('tooling-browser-export-surface.test: ok');
