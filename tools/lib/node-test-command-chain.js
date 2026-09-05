import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { delimiter, resolve } from 'node:path';

// Aggregate read-only checks from package.json, not a second command registry.
// Only simple npm chains are expanded. Shell programs, hooks and environment
// overrides retain npm's behavior and invalidate the in-invocation test set.
export function runNodeTestScripts(names, runTests, {
  root = process.cwd(),
  scripts = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8')).scripts,
  execute = (command, name) => spawnSync(command, {
    cwd: root,
    shell: true,
    stdio: 'inherit',
    env: {
      ...process.env,
      PATH: `${resolve(root, 'node_modules/.bin')}${delimiter}${process.env.PATH ?? ''}`,
      npm_lifecycle_event: name,
      npm_lifecycle_script: scripts[name],
    },
  }),
} = {}) {
  if (!names.length) throw new Error('Missing script names for --scripts');
  const seen = new Set();
  function run(command, name) {
    const result = execute(command, name);
    if (result.error || result.status !== 0) {
      throw new Error(`[node-tests] ${name} failed: ${result.error?.message ?? `exit ${result.status ?? 1}`}`);
    }
  }
  function visit(name, ancestors = []) {
    if (!/^[\w:.-]+$/.test(name) || !Object.hasOwn(scripts, name)) {
      throw new Error(`Unknown npm script: ${name}`);
    }
    if (ancestors.includes(name)) throw new Error(`Cyclic npm script: ${[...ancestors, name].join(' -> ')}`);
    const body = scripts[name];
    console.log(`[node-tests] script: ${name}`);
    // Do not parse shell syntax or bypass npm lifecycle hooks.
    if (/["'`\\;$()|<>\n]/.test(body) || scripts[`pre${name}`] || scripts[`post${name}`]) {
      seen.clear();
      run(`npm run ${name}`, name);
      seen.clear();
      return;
    }
    const commands = body.split(/\s+&&\s+/);
    if (commands.some((command) => /&/.test(command)
      || !/^(?:node |tsc |npm run [\w:.-]+$)/.test(command))) {
      seen.clear();
      run(`npm run ${name}`, name);
      seen.clear();
      return;
    }
    for (const command of commands) {
      const nested = /^npm run ([\w:.-]+)$/.exec(command);
      const tests = /^node tools\/run-node-tests\.js(?: ([-\w./= ]+))?$/.exec(command);
      if (nested) visit(nested[1], [...ancestors, name]);
      else if (tests) runTests(tests[1]?.split(/\s+/) ?? [], seen);
      else {
        // A separate checker, generator or formatter may change test inputs.
        // Reuse is safe only across consecutive isolated test batches.
        seen.clear();
        run(command, name);
        seen.clear();
      }
    }
  }
  for (const name of names) visit(name);
}
