import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import { resolveTestFiles } from '../../tools/lib/node-test-suites.js';
import { resolveTestFiles as coverageFiles } from '../../tools/run-node-coverage.js';
import { runNodeTestScripts } from '../../tools/lib/node-test-command-chain.js';

const root = await mkdtemp(join(tmpdir(), 'doppler-test-discovery-'));
try {
  for (const directory of ['tooling', 'runtime', 'pack', 'production-release', 'kernels']) {
    await mkdir(join(root, 'tests', directory), { recursive: true });
    await writeFile(join(root, 'tests', directory, 'sample.test.js'), '');
  }
  const pending = join(root, 'tests/runtime/feature.pending.test.js');
  await writeFile(pending, '');
  const unit = resolveTestFiles('unit', [], { root });
  assert.equal(unit.length, 4);
  assert.equal(resolveTestFiles('all', [], { root }).length, 5);
  assert.equal(resolveTestFiles('gpu', [], { root }).length, 1);
  assert.equal(resolveTestFiles('all', [], { root, includePending: true }).length, 6);
  assert.deepEqual(resolveTestFiles('unit', [pending], { root }), [pending]);
  assert.deepEqual(resolveTestFiles('all', ['tests', 'tests/tooling'], { root }), resolveTestFiles('all', [], { root }));
  assert.throws(() => resolveTestFiles('missing', [], { root }), /Unknown --suite/);
  assert.throws(() => resolveTestFiles('all', ['absent'], { root }), /Test path not found/);
  await writeFile(join(root, 'tests/README.md'), '');
  assert.throws(() => resolveTestFiles('all', ['tests/README.md'], { root }), /must end with/);
  assert.deepEqual(coverageFiles('unit', []), resolveTestFiles('unit', []));
  const listed = spawnSync(process.execPath, ['tools/run-node-tests.js', '--list', join(root, 'tests')], { encoding: 'utf8' });
  assert.equal(listed.status, 0, listed.stderr);
  assert.deepEqual(JSON.parse(listed.stdout), resolveTestFiles('all', ['tests'], { root }).map((file) => relative(process.cwd(), file)));

  const failedAssertion = join(root, 'async-failure.test.js');
  await writeFile(failedAssertion, `
    const { test } = require('node:test');
    test('asynchronous assertion must fail the launcher', async () => {
      await new Promise(setImmediate);
      throw new Error('async assertion failure');
    });
  `);
  const failed = spawnSync(process.execPath, ['tools/run-node-tests.js', failedAssertion], {
    encoding: 'utf8', env: { ...process.env, NODE_TEST_CONTEXT: 'child-v8' },
  });
  assert.equal(failed.status, 1, 'registered asynchronous tests must finish before success is reported');
  assert.match(failed.stdout + failed.stderr, /async assertion failure/);

  const passedAssertion = join(root, 'async-success.test.js');
  await writeFile(passedAssertion, `
    const { test } = require('node:test');
    setInterval(() => {}, 1000);
    test('asynchronous assertion completes before forced cleanup', async () => {
      await new Promise(setImmediate);
      console.log('async assertion completed');
    });
  `);
  const passed = spawnSync(process.execPath, ['tools/run-node-tests.js', passedAssertion], {
    encoding: 'utf8', timeout: 10000,
  });
  assert.equal(passed.status, 0, passed.stderr);
  assert.match(passed.stdout, /async assertion completed/);

  const plainRoot = join(root, 'plain');
  await mkdir(plainRoot);
  await writeFile(join(plainRoot, 'package.json'), '{"type":"module"}');
  const plainCases = [];
  for (const expectedStatus of [0, 1]) {
    const file = join(plainRoot, `assertion-${expectedStatus}.test.js`);
    const marker = `plain assertion ${expectedStatus} completed`;
    await writeFile(file, `
      import assert from 'node:assert/strict';
      assert.ok(globalThis.GPUBufferUsage, 'runtime setup must precede module evaluation');
      setInterval(() => {}, 1000);
      await new Promise(setImmediate);
      console.log('${marker}');
      ${expectedStatus === 1 ? "throw new Error('plain assertion failure');" : ''}
    `);
    const result = spawnSync(process.execPath, ['tools/run-node-tests.js', file], {
      encoding: 'utf8', timeout: 10000,
      env: { ...process.env, NODE_TEST_CONTEXT: 'child-v8' },
    });
    assert.equal(result.status, expectedStatus, result.stderr || result.stdout);
    assert.equal(result.stdout.split(marker).length - 1, 1, 'top-level assertions must execute exactly once');
    plainCases.push([file, expectedStatus, marker]);
  }

  const mixedAssertion = join(plainRoot, 'mixed-assertion.test.js');
  await writeFile(mixedAssertion, `
    import { test } from 'node:test';
    await test('registered test completes before module evaluation', () => {});
    await new Promise((resolve) => setTimeout(resolve, 50));
    throw new Error('late top-level assertion failure');
  `);
  const mixed = spawnSync(process.execPath, ['tools/run-node-tests.js', mixedAssertion], {
    encoding: 'utf8', timeout: 10000,
    env: { ...process.env, NODE_TEST_CONTEXT: 'child-v8' },
  });
  assert.equal(mixed.status, 1, mixed.stderr || mixed.stdout);
  assert.match(mixed.stdout + mixed.stderr, /late top-level assertion failure/);

  for (const [file, expectedStatus, marker] of [
    [failedAssertion, 1, 'async assertion failure'],
    [passedAssertion, 0, 'async assertion completed'],
    [mixedAssertion, 1, 'late top-level assertion failure'],
    ...plainCases,
  ]) {
    const covered = spawnSync(process.execPath, ['tools/run-node-coverage.js', '--no-threshold', file], {
      encoding: 'utf8', timeout: 10000,
      env: { ...process.env, NODE_TEST_CONTEXT: 'child-v8' },
    });
    assert.equal(covered.status, expectedStatus, covered.stderr || covered.stdout);
    assert.match(covered.stdout, /\[coverage\] all files/);
    assert.ok((covered.stdout + covered.stderr).includes(marker));
  }
} finally {
  await rm(root, { recursive: true, force: true });
}

const scripts = {
  check: 'npm run first && npm run second',
  first: 'node checker.js && node tools/run-node-tests.js tests/a.test.js',
  second: 'node tools/run-node-tests.js tests/a.test.js tests/b.test.js',
  complex: 'node -e "process.stdout.write(\'a && b\')"',
  hooked: 'node tools/run-node-tests.js tests/a.test.js',
  prehooked: 'node setup.js',
  cycle: 'npm run cycle',
  failed: 'node fail.js && npm run second',
  mutate: 'npm run second && node mutate.js && npm run second',
  directory: 'cd tests && npm run second',
  environment: 'export FLAG=value && npm run second',
  assignment: 'FLAG=value && npm run second',
};
const commands = [];
const tested = [];
const runTests = (files, seen) => {
  for (const file of files) if (!seen.has(file)) { tested.push(file); seen.add(file); }
};
const options = { scripts, execute: (command) => { commands.push(command); return { status: 0 }; } };
runNodeTestScripts(['check'], runTests, options);
assert.deepEqual(tested, ['tests/a.test.js', 'tests/b.test.js']);
assert.deepEqual(commands, ['node checker.js']);
runNodeTestScripts(['second'], runTests, options);
assert.equal(tested.length, 4, 'no success cache survives a separate invocation');
runNodeTestScripts(['first', 'complex', 'hooked', 'first'], runTests, options);
assert.ok(commands.includes('npm run complex'));
assert.ok(commands.includes('npm run hooked'));
assert.equal(tested.length, 6, 'opaque shell scripts invalidate prior success');
assert.throws(() => runNodeTestScripts([], runTests, options), /Missing script names/);
assert.throws(() => runNodeTestScripts(['absent'], runTests, options), /Unknown npm script/);
assert.throws(() => runNodeTestScripts(['cycle'], runTests, options), /Cyclic npm script/);
let attempted = 0;
assert.throws(() => runNodeTestScripts(['failed'], () => { attempted += 1; }, {
  scripts, execute: () => ({ status: 7 }),
}), /exit 7/);
assert.equal(attempted, 0, 'a failed prerequisite stops subsequent tests');
const beforeMutation = tested.length;
runNodeTestScripts(['mutate'], runTests, options);
assert.equal(tested.length - beforeMutation, 4, 'tests run again after potentially changed inputs');
const beforeShell = tested.length;
runNodeTestScripts(['directory', 'environment', 'assignment'], runTests, options);
assert.equal(tested.length, beforeShell, 'stateful chains are not split into parent-context tests');
assert.deepEqual(commands.slice(-3), ['npm run directory', 'npm run environment', 'npm run assignment']);
console.log('node-test-discovery.test: ok');
