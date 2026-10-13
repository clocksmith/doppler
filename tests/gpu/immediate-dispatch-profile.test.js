import assert from 'node:assert/strict';
import { beginImmediateDispatchProfile, dispatch, dispatchIndirect } from '../../src/gpu/kernels/dispatch.js';

const events = [];
const device = { limits: { maxComputeWorkgroupsPerDimension: 65535 },
  createCommandEncoder() { throw Error('unprofiled encoder'); } };
function makeRecorder() {
  return { device, isProfilingEnabled: () => true,
    recordDispatch(pipeline, bindings, workgroups, label) { events.push(['direct', pipeline, bindings, workgroups, label]); },
    recordDispatchIndirect(pipeline, bindings, buffer, offset, label) { events.push(['indirect', pipeline, bindings, buffer, offset, label]); },
    submitPartial() { events.push(['submit']); } };
}
let recorder = makeRecorder();
const pipeline = { label: 'paired_q4k' }, bindings = {}, indirect = {};
const release = beginImmediateDispatchProfile(device, () => recorder);
assert.throws(() => beginImmediateDispatchProfile(device, () => recorder), /already leased/);
dispatch(device, pipeline, bindings, [65535, 15, 1], 'matmul');
dispatchIndirect(device, pipeline, bindings, indirect, 16, 'indirect');
assert.deepEqual(events.map(row => row[0]), ['direct', 'submit', 'indirect', 'submit']);
assert.equal(events[0][1], pipeline);
assert.equal(events[0][2], bindings);
assert.deepEqual(events[0][3], [65535, 15, 1]);
assert.match(events[0][4], /paired_q4k/);
assert.match(events[0][4], /65535,15,1/);
assert.equal(events[2][4], 16);
// Rotation follows the existing layer barrier; the previous recorder is never reused.
recorder = makeRecorder();
dispatch(device, pipeline, bindings, 3);
assert.deepEqual(events.at(-2)[3], [3, 1, 1]);
const count = events.length;
assert.throws(() => dispatch(device, pipeline, bindings, [65536, 1, 1]), /exceeds/);
assert.equal(events.length, count);
recorder = { ...makeRecorder(), isProfilingEnabled: () => false };
assert.throws(() => dispatch(device, pipeline, bindings, 1), /timestamp/);
release(); release();
assert.throws(() => dispatch(device, pipeline, bindings, 1), /unprofiled encoder/);
assert.throws(() => beginImmediateDispatchProfile(device, () => recorder), /timestamp/);
recorder = { ...makeRecorder(), device: {} };
assert.throws(() => beginImmediateDispatchProfile(device, () => recorder), /device/);
recorder = makeRecorder();
const releaseAgain = beginImmediateDispatchProfile(device, () => recorder);
recorder.submitPartial = () => { throw Error('submit rejected'); };
assert.throws(() => dispatch(device, pipeline, bindings, 1), /submit rejected/);
releaseAgain();
recorder = makeRecorder();
const getter = () => recorder;
const oldRelease = beginImmediateDispatchProfile(device, getter);
oldRelease();
const newRelease = beginImmediateDispatchProfile(device, getter);
oldRelease();
assert.throws(() => beginImmediateDispatchProfile(device, getter), /already leased/,
  'a stale release cannot remove a new lease even with the same getter');
newRelease();
console.log('immediate-dispatch-profile: pass');
