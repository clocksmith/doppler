// Execute the actual diagnostic orchestration with a deferred operation,
// replacing only numerical work and shortening timers for CPU failure injection.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const source = await readFile(new URL('./linear-control.js', import.meta.url), 'utf8');
const definitions = source.slice(source.indexOf('const events = [];'), source.indexOf('const persist ='));
const start = source.indexOf('try {\n const operation = async () => {');
const bodyStart = source.indexOf('async () => {', start) + 'async () => {'.length;
const bodyEnd = source.indexOf('\n };\n operationPromise', bodyStart);
assert(start >= 0 && bodyEnd > bodyStart);
const tail = source.slice(start, bodyStart) + '\n await work(); guard(); mark("dispatch"); receipt.passed = true;' + source.slice(bodyEnd, source.indexOf('\nconsole.log(JSON.stringify({ passed:'));
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const run = new AsyncFunction('receipt', 'host', 'work', 'mark', 'persist', 'process',
  definitions.replaceAll('10000', '20') + '\nlet writes = Promise.resolve();\n' + tail.replaceAll('300000', '5'));
for (const delay of [0, 10, 50]) {
  const receipt = {}, events = [];
  const host = { close: async () => { events.push('host-close'); } };
  const work = () => new Promise(resolve => setTimeout(resolve, delay));
  await run(receipt, host, work, phase => events.push(phase), async () => {}, { exitCode: 0 });
  if (delay === 0) {
    assert.equal(receipt.passed, true); assert(events.indexOf('dispatch') < events.indexOf('host-close'));
  } else {
    assert.equal(receipt.passed, false); assert.match(receipt.error, /Operation exceeded/); assert(!events.includes('dispatch'));
    if (delay === 50) { assert.equal(receipt.hostClosed, undefined); assert.equal(receipt.ownership, 'unknown'); assert(!events.includes('host-close')); }
    await new Promise(resolve => setTimeout(resolve, delay + 5));
    assert(!events.includes('dispatch'), 'Late completion must not submit work'); assert.equal(receipt.passed, false, 'Late completion must not overwrite timeout');
  }
}
console.log('actual diagnostic orchestration: success, late-operation cancellation, and unknown cleanup pass');
