import assert from 'node:assert/strict';
import { PreparedDecodeWorkspace } from '../../src/gpu/prepared-decode-workspace.js';

globalThis.GPUBufferUsage = { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 };
let destroyed = 0, created = 0, bound = 0, fail = false;
const device = {
  createBuffer({ size }) {
    if (fail) throw new Error('allocation failure');
    created++;
    return { size, destroy() { destroyed++; } };
  },
  createBindGroup() { bound++; return {}; },
};
const owner = new PreparedDecodeWorkspace(device);
const a = owner.getBuffer('L0:q', 64), b = owner.getBuffer('L1:q', 64);
assert.equal(owner.getBuffer('L0:q', 64), a);
assert.notEqual(a, b, 'different layers cannot share live projection storage');
assert.throws(() => owner.getBuffer('L0:q', 128), /changed size/);
fail = true;
assert.throws(() => owner.getBuffer('new', 64), /allocation failure/);
fail = false;
assert.equal(owner.getStats().buffers, 2, 'failed allocation never becomes an owned slot');
const layout = {}, descriptor = { layout, entries: [{ binding: 0, resource: { buffer: a, offset: 0, size: 32 } }] };
const first = owner.getBindGroup('q', descriptor);
assert.equal(owner.getBindGroup('q', descriptor), first);
assert.equal(bound, 1);
descriptor.entries[0].resource.size = 64;
assert.notEqual(owner.getBindGroup('q', descriptor), first, 'range changes cannot reuse a binding');
descriptor.entries[0].resource.buffer = b;
owner.getBindGroup('q', descriptor);
owner.getBindGroup('q', { ...descriptor, layout: {} });
assert.equal(bound, 4, 'buffers and layouts participate in exact binding identity');
assert.equal(owner.ownsBuffer(a), true);
assert.equal(owner.ownsBuffer({}), false);
const secondOwner = new PreparedDecodeWorkspace(device);
assert.notEqual(secondOwner.getBuffer('L0:q', 64), a, 'sessions cannot alias scratch');
owner.release(); owner.release();
assert.equal(destroyed, 2);
assert.equal(owner.getStats().bindings, 0);
assert.throws(() => owner.getBuffer('q', 64), /released/);
assert.throws(() => owner.getBindGroup('q', descriptor), /released/);
secondOwner.release();
assert.equal(created, destroyed);
let lose;
const lostDevice = { ...device, lost: new Promise(resolve => { lose = resolve; }) };
const lostOwner = new PreparedDecodeWorkspace(lostDevice);
const live = lostOwner.getBuffer('q', 64);
lose({ reason: 'unknown', message: 'injected loss' });
await Promise.resolve();
assert.throws(() => lostOwner.getBuffer('q', 64), /lost/);
assert.throws(() => lostOwner.getBindGroup('q', { layout, entries: [] }), /lost/);
assert.equal(lostOwner.ownsBuffer(live), true, 'loss does not discard the cleanup owner');
lostOwner.release();
assert.equal(created, destroyed);
console.log('prepared-decode-workspace: passed ownership, identity, failure and device-loss checks');
