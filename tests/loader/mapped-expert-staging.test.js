import assert from 'node:assert/strict';
import test from 'node:test';

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 };
globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 };
const { getMappedExpertStaging, destroyMappedExpertStaging, stageRoutedExperts, beginStagedRelease, settleStagedRelease, assertStagedVerification } = await import('../../src/loader/experts/mapped-expert-staging.js');
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
    queue: { submit(buffers) { log.submits++; log.onSubmit?.(); for (const copies of buffers) for (const [s, at, t, to, n] of copies) t.data.set(s.data.subarray(at, at + n), to); },
      async onSubmittedWorkDone() {} } };
}
// Two planes of 8 and 4 bytes per expert, stored back to back per plane in shards 0 and 1.
const EXPERTS = 6;
const source = [Uint8Array.from({ length: EXPERTS * 8 }, (_, i) => i + 1), Uint8Array.from({ length: EXPERTS * 4 }, (_, i) => 200 - i)];
const plan = { layer: 0, numExperts: EXPERTS, expertBytes: 12, planes: [
  { role: 'gate', expertBytes: 8, location: { shardIndex: 0, offset: 0, size: EXPERTS * 8 } },
  { role: 'down', expertBytes: 4, location: { shardIndex: 1, offset: 0, size: EXPERTS * 4 } }] };
function fixture({ verified = true, readMany = true, readAhead = true, verifies = true, policy = {} } = {}) {
  // `events` records the order of reads ahead of verification, lane verifications and submits.
  const state = { verified, reads: 0, aheadReads: 0, verifications: 0, fail: null, events: [], gate: null };
  const fill = (shard, offset, destination) => destination.set(source[shard].subarray(offset, offset + destination.byteLength));
  const port = { maxPartBytes: 64, maxRanges: 3,
    async readVerifiedInto(shard, offset, destination) {
      if (state.fail) throw state.fail;
      if (!state.verified) return false;
      state.reads++; fill(shard, offset, destination); return true;
    },
    ...(readAhead ? { hasVerified: () => state.verified,
      async readAheadInto(shard, offset, destination) {
        if (state.fail) throw state.fail;
        if (state.verified) { state.reads++; fill(shard, offset, destination); return true; }
        state.aheadReads++; state.events.push('read-ahead'); fill(shard, offset, destination);
        return () => state.verified;
      } } : {}),
    ...(readMany ? { async readMany(ranges) {
      state.events.push('verify-start'); await state.gate;
      state.verifications += ranges.length; if (verifies) state.verified = true;
      state.events.push('verify-end');
      return { bytes: new Uint8Array(1), timings: { workerWallMs: 4, readWaitMs: 1, verificationMs: 2, copyMs: 0, roundTripMs: 5 }, release() {} };
    },
    // Background verification: the same lane job, queued, with the lane's bytes dropped.
    async verifyMany(ranges) {
      state.events.push('background-start'); await state.gate;
      if (state.reject) throw state.reject;
      state.verifications += ranges.length; if (verifies) state.verified = true;
      state.events.push('background-end');
      return { workerWallMs: 4, readWaitMs: 1, verificationMs: 2, copyMs: 0, roundTripMs: 5 };
    } } : {}) };
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

test('an unverified range is read while a lane verifies it, and reaches the model only after the lane has finished', async () => {
  const f = fixture({ verified: false }), staging = getMappedExpertStaging(f.options, plan);
  const release = Promise.withResolvers(); f.state.gate = release.promise;
  f.dev.log.onSubmit = () => f.state.events.push('submit');
  const upload = stageRoutedExperts(staging, f.stats, plan, [1], f.targetOf, f.signal, null, null);
  // Both reads have run while the lane is still verifying, and nothing has been submitted.
  await new Promise(resolve => setTimeout(resolve, 5));
  assert.deepEqual(f.state.events, ['read-ahead', 'read-ahead', 'verify-start']);
  assert.equal(f.dev.log.submits, 0);
  assert(f.targets.gate.data.every(byte => byte === 0), 'unverified bytes are not in a model buffer');
  release.resolve();
  assert.equal(await upload, true);
  assert.deepEqual(f.state.events, ['read-ahead', 'read-ahead', 'verify-start', 'verify-end', 'submit']);
  // Each range was read once: no second read after verification.
  assert.deepEqual([f.state.aheadReads, f.state.reads, f.state.verifications, f.dev.log.submits], [2, 0, 2, 1]);
  assert.deepEqual([...f.targets.gate.data.subarray(8, 16)], [...source[0].subarray(8, 16)]);
  assert.deepEqual([...f.targets.down.data.subarray(4, 8)], [...source[1].subarray(4, 8)]);
  // The lane's job is timed, and the bytes read ahead are counted.
  assert.deepEqual([f.stats.sourceTimingJobs, f.stats.sourceUntimedJobs, f.stats.sourceJobVerificationMs, f.stats.stagedReadAheadBytes], [1, 0, 2, 12]);
  assert(f.stats.stagedVerifyWaitMs >= f.stats.stagedReadWaitMs && f.stats.stagedReadWaitMs >= 0);
  // Verified now: the next upload reads directly and runs no lane.
  assert.equal(await stageRoutedExperts(staging, f.stats, plan, [2], f.targetOf, f.signal, null, null), true);
  assert.deepEqual([f.state.aheadReads, f.state.reads, f.state.verifications], [2, 2, 2]);
});

test('bytes read ahead are discarded when verification does not confirm them, fails, or is unavailable', async () => {
  // The lane finishes without the pieces becoming verified (for example the file state changed).
  const unconfirmed = fixture({ verified: false, verifies: false }), a = getMappedExpertStaging(unconfirmed.options, plan);
  assert.equal(await stageRoutedExperts(a, unconfirmed.stats, plan, [1], unconfirmed.targetOf, unconfirmed.signal, null, null), false);
  assert.deepEqual([unconfirmed.dev.log.submits, unconfirmed.stats.uploadBytes, unconfirmed.stats.sourceReads], [0, 0, 0]);
  assert(unconfirmed.targets.gate.data.every(byte => byte === 0) && unconfirmed.targets.down.data.every(byte => byte === 0));
  // The slot went back to the ring mapped, and serves a later verified upload.
  unconfirmed.state.verified = true;
  assert.equal(await stageRoutedExperts(a, unconfirmed.stats, plan, [1], unconfirmed.targetOf, unconfirmed.signal, null, null), true);

  // The lane rejects the pieces (a hash mismatch): the upload fails and nothing is submitted.
  const forged = fixture({ verified: false }), b = getMappedExpertStaging(forged.options, plan);
  forged.port.readMany = async () => { throw new Error('Piece identity mismatch.'); };
  const rebound = getMappedExpertStaging({ ...forged.options, rangeLease: { ...forged.port } }, plan);
  await assert.rejects(stageRoutedExperts(rebound, forged.stats, plan, [1], forged.targetOf, forged.signal, null, null), /identity mismatch/);
  assert.equal(forged.dev.log.submits, 0);
  assert(forged.targets.gate.data.every(byte => byte === 0));
  assert(b);

  // No lane, or a port that cannot read ahead: declined untouched, nothing read into the slot.
  for (const options of [{ verified: false, readMany: false }, { verified: false, readAhead: false }]) {
    const g = fixture(options), declined = getMappedExpertStaging(g.options, plan);
    assert.equal(await stageRoutedExperts(declined, g.stats, plan, [1], g.targetOf, g.signal, null, null), false);
    assert.deepEqual([g.dev.log.submits, g.stats.uploadBytes, g.stats.sourceReads, g.state.aheadReads], [0, 0, 0, 0]);
    assert(g.targets.gate.data.every(byte => byte === 0), 'a declined upload writes nothing to its target');
  }
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

const deferring = { uploadVerification: 'before-release' };

test('inside a release gate, first-time bytes are submitted at once and the gate waits for their hashes', async () => {
  const f = fixture({ verified: false, policy: deferring }), staging = getMappedExpertStaging(f.options, plan);
  const release = Promise.withResolvers(); f.state.gate = release.promise;
  f.dev.log.onSubmit = () => f.state.events.push('submit');
  beginStagedRelease(f.port);
  assert.equal(await stageRoutedExperts(staging, f.stats, plan, [1], f.targetOf, f.signal, null, null), true);
  // The bytes are in the model buffers and the lane has not finished.
  assert.deepEqual(f.state.events, ['read-ahead', 'read-ahead', 'submit', 'background-start']);
  assert.deepEqual([...f.targets.gate.data.subarray(8, 16)], [...source[0].subarray(8, 16)]);
  assert.deepEqual([f.stats.stagedDeferredBytes, f.stats.sourceTimingJobs, f.stats.uploadBytes], [12, 0, 12]);
  // The gate does not settle until the lane has.
  let settled = false;
  const settling = settleStagedRelease(f.port).then(() => { settled = true; });
  await new Promise(resolve => setTimeout(resolve, 5));
  assert.equal(settled, false);
  release.resolve();
  await settling;
  assert.deepEqual(f.state.events.slice(-1), ['background-end']);
  assert.deepEqual([f.stats.sourceTimingJobs, f.stats.sourceJobVerificationMs], [1, 2]);
  assert(f.stats.stagedReleaseWaitMs > 0);
  // The next step opens normally, and verified ranges are not deferred.
  beginStagedRelease(f.port);
  assert.equal(await stageRoutedExperts(staging, f.stats, plan, [2], f.targetOf, f.signal, null, null), true);
  await settleStagedRelease(f.port);
  assert.equal(f.stats.stagedDeferredBytes, 12);
  await destroyMappedExpertStaging(f.options);
});

test('a hash that fails after upload fails the step and every later step and upload', async () => {
  for (const mode of ['rejected', 'unconfirmed']) {
    const f = fixture({ verified: false, policy: deferring, verifies: mode !== 'unconfirmed' }), staging = getMappedExpertStaging(f.options, plan);
    if (mode === 'rejected') f.state.reject = new Error('Piece identity mismatch.');
    beginStagedRelease(f.port);
    assert.equal(await stageRoutedExperts(staging, f.stats, plan, [1], f.targetOf, f.signal, null, null), true);
    await assert.rejects(settleStagedRelease(f.port), mode === 'rejected' ? /verification failed after upload: Piece identity mismatch/ : /could not be confirmed verified/);
    // Unverified bytes are resident: no later step may begin and nothing more is staged.
    assert.throws(() => beginStagedRelease(f.port), /verification failed after upload/);
    await assert.rejects(stageRoutedExperts(staging, f.stats, plan, [2], f.targetOf, f.signal, null, null), /verification failed after upload/);
    assert.equal(f.dev.log.submits, 1);
    await destroyMappedExpertStaging(f.options);
  }
  // A step that already failed for another reason still waits for its hashes, and keeps a hash failure.
  const g = fixture({ verified: false, policy: deferring }), staging = getMappedExpertStaging(g.options, plan);
  g.state.reject = new Error('Piece identity mismatch.');
  beginStagedRelease(g.port);
  assert.equal(await stageRoutedExperts(staging, g.stats, plan, [1], g.targetOf, g.signal, null, null), true);
  await settleStagedRelease(g.port, { release: false });
  assert.throws(() => beginStagedRelease(g.port), /verification failed after upload/);
  await assert.rejects(settleStagedRelease(g.port), /gate is not open/);
});

test('nothing is deferred outside a release gate, under the default policy, or without background verification', async () => {
  const cases = [
    { name: 'no gate', options: { verified: false, policy: deferring }, gate: false },
    { name: 'default policy', options: { verified: false }, gate: true },
    { name: 'no background verification', options: { verified: false, policy: deferring }, gate: true, strip: true }];
  for (const item of cases) {
    const f = fixture(item.options);
    if (item.strip) delete f.port.verifyMany;
    const staging = getMappedExpertStaging(f.options, plan);
    f.dev.log.onSubmit = () => f.state.events.push('submit');
    if (item.gate) beginStagedRelease(f.port);
    assert.equal(await stageRoutedExperts(staging, f.stats, plan, [1], f.targetOf, f.signal, null, null), true, item.name);
    assert.deepEqual(f.state.events, ['read-ahead', 'read-ahead', 'verify-start', 'verify-end', 'submit'], item.name);
    assert.equal(f.stats.stagedDeferredBytes, 0, item.name);
    if (item.gate) await settleStagedRelease(f.port);
  }
});

test('an upload that has the lanes to itself spreads its hashing across them; a crowded one keeps one job', async () => {
  // Four lanes and one upload in flight: its two ranges are verified by two jobs at the same time.
  const alone = fixture({ verified: false }); alone.port.concurrency = 4;
  const staging = getMappedExpertStaging(alone.options, plan);
  assert.equal(await stageRoutedExperts(staging, alone.stats, plan, [1], alone.targetOf, alone.signal, null, null), true);
  assert.deepEqual(alone.state.events.filter(event => event.startsWith('background')), ['background-start', 'background-start', 'background-end', 'background-end']);
  assert.deepEqual([alone.state.verifications, alone.stats.sourceTimingJobs, alone.dev.log.submits], [2, 2, 1]);
  assert.deepEqual([...alone.targets.gate.data.subarray(8, 16)], [...source[0].subarray(8, 16)]);
  // One lane: one job for the whole upload, through the foreground path, as before.
  const single = fixture({ verified: false });
  const one = getMappedExpertStaging(single.options, plan);
  assert.equal(await stageRoutedExperts(one, single.stats, plan, [1], single.targetOf, single.signal, null, null), true);
  assert.deepEqual(single.state.events, ['read-ahead', 'read-ahead', 'verify-start', 'verify-end']);
  // A job that fails still fails the upload after its sibling has settled, and nothing is submitted.
  const broken = fixture({ verified: false }); broken.port.concurrency = 4; broken.state.reject = new Error('Piece identity mismatch.');
  const two = getMappedExpertStaging(broken.options, plan);
  await assert.rejects(stageRoutedExperts(two, broken.stats, plan, [1], broken.targetOf, broken.signal, null, null), /identity mismatch/);
  assert.equal(broken.dev.log.submits, 0);
  assert(broken.targets.gate.data.every(byte => byte === 0));
});

test('each range needs its own confirmation: one unconfirmed range among several keeps the whole upload out', async () => {
  // Per-range state: the gate plane becomes verified, the down plane never does.
  const f = fixture({ verified: false });
  const done = new Set();
  f.port.hasVerified = shard => done.has(shard);
  f.port.readAheadInto = async (shard, offset, destination) => {
    destination.set(source[shard].subarray(offset, offset + destination.byteLength));
    return done.has(shard) ? true : () => done.has(shard);
  };
  f.port.readVerifiedInto = async (shard, offset, destination) => {
    if (!done.has(shard)) return false;
    destination.set(source[shard].subarray(offset, offset + destination.byteLength)); return true;
  };
  // The staging ring binds the port's methods when it is created, so the lane's behaviour is switched by a flag.
  let lanesVerifyEverything = false;
  f.port.readMany = async ranges => {
    for (const range of ranges) if (lanesVerifyEverything || range.shard === 0) done.add(range.shard);
    return { bytes: new Uint8Array(1), release() {} };
  };
  const staging = getMappedExpertStaging(f.options, plan);
  assert.equal(await stageRoutedExperts(staging, f.stats, plan, [1], f.targetOf, f.signal, null, null), false);
  assert.equal(f.dev.log.submits, 0);
  assert(f.targets.gate.data.every(byte => byte === 0) && f.targets.down.data.every(byte => byte === 0));
  // Mixed: the gate plane is now verified and read directly, the down plane is read ahead and then confirmed.
  lanesVerifyEverything = true;
  assert.equal(await stageRoutedExperts(staging, f.stats, plan, [1], f.targetOf, f.signal, null, null), true);
  assert.deepEqual([...f.targets.gate.data.subarray(8, 16)], [...source[0].subarray(8, 16)]);
  assert.deepEqual([...f.targets.down.data.subarray(4, 8)], [...source[1].subarray(4, 8)]);
});

test('deferral needs an open gate at submit time, a usable read result, and survives a broken counter', async () => {
  // The gate closes while the bytes are being read: nothing is submitted.
  const closing = fixture({ verified: false, policy: deferring });
  const reading = Promise.withResolvers(), readAhead = closing.port.readAheadInto;
  closing.port.readAheadInto = async (...args) => { await reading.promise; return readAhead(...args); };
  const a = getMappedExpertStaging(closing.options, plan);
  beginStagedRelease(closing.port);
  const late = stageRoutedExperts(a, closing.stats, plan, [1], closing.targetOf, closing.signal, null, null);
  await new Promise(resolve => setTimeout(resolve, 5));   // the upload has chosen to defer and is reading
  await settleStagedRelease(closing.port);
  reading.resolve();
  assert.equal(await late, false);
  assert.equal(closing.dev.log.submits, 0);

  // A read that reports neither "verified" nor a confirmation to run is not submitted.
  const vague = fixture({ verified: false, policy: deferring });
  vague.port.readAheadInto = async (shard, offset, destination) => { destination.fill(7); return undefined; };
  const b = getMappedExpertStaging(vague.options, plan);
  beginStagedRelease(vague.port);
  assert.equal(await stageRoutedExperts(b, vague.stats, plan, [1], vague.targetOf, vague.signal, null, null), false);
  await settleStagedRelease(vague.port);
  assert.equal(vague.dev.log.submits, 0);

  // A lane job whose timings cannot be recorded has still verified its bytes: the gate does not fail.
  const odd = fixture({ verified: false, policy: deferring });
  odd.port.verifyMany = async () => { odd.state.verified = true; return { workerWallMs: 'not a number' }; };
  const c = getMappedExpertStaging(odd.options, plan);
  beginStagedRelease(odd.port);
  assert.equal(await stageRoutedExperts(c, odd.stats, plan, [1], odd.targetOf, odd.signal, null, null), true);
  await settleStagedRelease(odd.port);
  assertStagedVerification(odd.port);
  beginStagedRelease(odd.port); await settleStagedRelease(odd.port);
});

test('after a failed deferred verification every request for experts on that port is refused', async () => {
  const f = fixture({ verified: false, policy: deferring }), staging = getMappedExpertStaging(f.options, plan);
  assertStagedVerification(f.port);
  f.state.reject = new Error('Piece identity mismatch.');
  beginStagedRelease(f.port);
  assert.equal(await stageRoutedExperts(staging, f.stats, plan, [1], f.targetOf, f.signal, null, null), true);
  await assert.rejects(settleStagedRelease(f.port), /verification failed after upload/);
  assert.throws(() => assertStagedVerification(f.port), /verification failed after upload/);
  // A port that never deferred anything is unaffected.
  assertStagedVerification({});
});
