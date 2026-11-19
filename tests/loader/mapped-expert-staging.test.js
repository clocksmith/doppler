import assert from 'node:assert/strict';
import test from 'node:test';

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 };
globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 };
const { getMappedExpertStaging, destroyMappedExpertStaging, stageRoutedExperts } = await import('../../src/loader/experts/mapped-expert-staging.js');
const { createRoutedExpertCounters } = await import('../../src/loader/experts/routed-expert-source.js');

// A recording device: copies move real bytes between fake buffers at submit, mapping follows WebGPU's states.
function device() {
  const log = { created: [], submits: 0 };
  const make = ({ size, usage, mappedAtCreation }) => {
    const buffer = { size, usage, data: new Uint8Array(size), mapped: !!mappedAtCreation, destroyed: false,
      getMappedRange() { assert(buffer.mapped, 'getMappedRange on an unmapped buffer'); return buffer.data.buffer; },
      unmap() { assert(buffer.mapped); buffer.mapped = false; },
      async mapAsync() { assert(!buffer.mapped && !buffer.destroyed); await null; buffer.mapped = true; },
      destroy() { buffer.destroyed = true; } };
    log.created.push(buffer); return buffer;
  };
  return { log, createBuffer: make,
    createCommandEncoder: () => { const copies = []; return {
      copyBufferToBuffer(source, at, target, targetOffset, length) { assert(!source.mapped, 'copy from a mapped buffer'); copies.push([source, at, target, targetOffset, length]); },
      finish: () => copies }; },
    queue: { submit(buffers) { log.submits++; for (const copies of buffers) for (const [s, at, t, to, n] of copies) t.data.set(s.data.subarray(at, at + n), to); },
      async onSubmittedWorkDone() {} } };
}
// Two planes of 8 and 4 bytes per expert, stored back to back per plane in shards 0 and 1.
const EXPERTS = 6;
const source = [Uint8Array.from({ length: EXPERTS * 8 }, (_, i) => i + 1), Uint8Array.from({ length: EXPERTS * 4 }, (_, i) => 200 - i)];
const plan = { layer: 0, numExperts: EXPERTS, expertBytes: 12, planes: [
  { role: 'gate', expertBytes: 8, location: { shardIndex: 0, offset: 0, size: EXPERTS * 8 } },
  { role: 'down', expertBytes: 4, location: { shardIndex: 1, offset: 0, size: EXPERTS * 4 } }] };
function fixture({ verified = true, readMany = true, policy = {} } = {}) {
  const state = { verified, reads: 0, verifications: 0, fail: null };
  const port = { maxPartBytes: 64, maxRanges: 3,
    async readVerifiedInto(shard, offset, destination) {
      if (state.fail) throw state.fail;
      if (!state.verified) return false;
      state.reads++; destination.set(source[shard].subarray(offset, offset + destination.byteLength)); return true;
    },
    ...(readMany ? { async readMany(ranges) { state.verifications += ranges.length; state.verified = true; return { bytes: new Uint8Array(1), release() {} }; } } : {}) };
  const dev = device();
  const options = { device: dev, rangeLease: port, policy: { uploadStaging: 'mapped', sourceReadMode: 'expert-lease-batch', stagingBytes: 48, ...policy } };
  const targets = { gate: dev.createBuffer({ size: EXPERTS * 8, usage: 0 }), down: dev.createBuffer({ size: EXPERTS * 4, usage: 0 }) };
  const targetOf = (expert, plane) => ({ buffer: targets[plane.role], offset: expert * plane.expertBytes });
  return { state, port, dev, options, targets, targetOf, stats: createRoutedExpertCounters(), signal: new AbortController().signal };
}

test('staged experts reach their target planes in one submit and are counted as direct reads', async () => {
  const f = fixture(), staging = getMappedExpertStaging(f.options, plan);
  assert(staging); assert.equal(getMappedExpertStaging(f.options, plan), staging, 'one ring per port');
  assert.deepEqual([staging.slotBytes, staging.slotCount], [12, 4]);
  let hooks = 0;
  assert.equal(await stageRoutedExperts(staging, f.stats, plan, [4], f.targetOf, f.signal, () => hooks++, null), true);
  assert.deepEqual([...f.targets.gate.data.subarray(32, 40)], [...source[0].subarray(32, 40)]);
  assert.deepEqual([...f.targets.down.data.subarray(16, 20)], [...source[1].subarray(16, 20)]);
  assert.deepEqual([f.dev.log.submits, hooks, f.state.reads], [1, 1, 2]);
  assert.deepEqual([f.stats.sourceReads, f.stats.sourceReadBytes, f.stats.sourceReadIntoBytes, f.stats.uploadBytes, f.stats.sourceUntimedJobs], [2, 12, 12, 12, 1]);
  // The ring recycles its slots: many more uploads than slots, never more than slotCount buffers.
  for (let round = 0; round < 12; round++) assert.equal(await stageRoutedExperts(staging, f.stats, plan, [round % EXPERTS], f.targetOf, f.signal, null, null), true);
  assert.deepEqual([...f.targets.gate.data], [...source[0]]);
  assert(f.dev.log.created.filter(buffer => buffer.usage === (GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC)).length <= 4);
  await destroyMappedExpertStaging(f.options);
  assert(f.dev.log.created.filter(buffer => buffer.usage !== 0).every(buffer => buffer.destroyed));
});

test('an unverified range is authenticated by a lane once and then staged; without a lane it is declined untouched', async () => {
  const f = fixture({ verified: false }), staging = getMappedExpertStaging(f.options, plan);
  assert.equal(await stageRoutedExperts(staging, f.stats, plan, [1], f.targetOf, f.signal, null, null), true);
  assert.deepEqual([f.state.verifications, f.dev.log.submits], [2, 1]);
  assert.deepEqual([...f.targets.gate.data.subarray(8, 16)], [...source[0].subarray(8, 16)]);
  const g = fixture({ verified: false, readMany: false }), declined = getMappedExpertStaging(g.options, plan);
  assert.equal(await stageRoutedExperts(declined, g.stats, plan, [1], g.targetOf, g.signal, null, null), false);
  assert.deepEqual([g.dev.log.submits, g.stats.uploadBytes, g.stats.sourceReads], [0, 0, 0]);
  assert(g.targets.gate.data.every(byte => byte === 0), 'a declined upload writes nothing to its target');
});

test('a failed read submits nothing and leaves the ring usable; unsupported shapes and policies decline', async () => {
  const f = fixture(), staging = getMappedExpertStaging(f.options, plan);
  f.state.fail = new Error('Gather source changed during a direct verified read.');
  await assert.rejects(stageRoutedExperts(staging, f.stats, plan, [2], f.targetOf, f.signal, null, null), /changed during/);
  assert.equal(f.dev.log.submits, 0);
  f.state.fail = null;
  assert.equal(await stageRoutedExperts(staging, f.stats, plan, [2], f.targetOf, f.signal, null, null), true);
  // More experts than a slot holds, or a transformed plane, are left to the lease path.
  assert.equal(await stageRoutedExperts(staging, f.stats, plan, [0, 1], f.targetOf, f.signal, null, null), false);
  const transformed = { ...plan, planes: [{ ...plan.planes[0], location: { ...plan.planes[0].location, sourceTransform: {} } }, plan.planes[1]] };
  assert.equal(await stageRoutedExperts(staging, f.stats, transformed, [0], f.targetOf, f.signal, null, null), false);
  await assert.rejects(stageRoutedExperts(staging, f.stats, plan, [3], f.targetOf, AbortSignal.abort(new Error('stop')), null, null), /stop/);
  for (const policy of [{ uploadStaging: 'queue' }, { sourceReadMode: 'lease-parallel' }, { stagingBytes: 8 }]) {
    assert.equal(getMappedExpertStaging(fixture({ policy }).options, plan), null);
  }
  const noDirect = fixture(); delete noDirect.port.readVerifiedInto;
  assert.equal(getMappedExpertStaging(noDirect.options, plan), null);
});
