import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// Lifecycle-only mocks; physical numerical evidence is in speculative-prefix-recovery-gpu.test.js.
globalThis.GPUBufferUsage = { COPY_SRC: 4, COPY_DST: 8 };
globalThis.GPUBuffer = class { size = 16; usage = 12; };
let allocations = 0, failAt = Infinity;
const moduleURL = new URL('../../src/gpu/kernels/linear-attention-recovery.js', import.meta.url).href;
globalThis.recoveryOwnershipProbe = { acquire() {
  if (++allocations === failAt) throw Error('allocation failure');
  return new GPUBuffer();
} };
const hook = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === moduleURL && specifier === '../../memory/buffer-pool.js') {
    return { url: 'data:text/javascript,export const acquireBuffer = () => recoveryOwnershipProbe.acquire();', shortCircuit: true };
  }
  return next(specifier, context);
} });
const device = {};
const recorder = () => {
  const owned = new Set(), copies = [];
  return { device, submitted: false, owned, copies,
    getStats() { return { submitted: this.submitted }; },
    trackTemporaryBuffer(buffer) { owned.add(buffer); },
    getEncoder() { return { copyBufferToBuffer: (...args) => copies.push(args) }; },
    abort() { this.submitted = true; owned.clear(); },
  };
};
const state = () => ({ sourceLayout: 'gguf-v1', recurrentStatePrecision: 'f32',
  convStateGPU: new GPUBuffer(), recurrentStateGPU: new GPUBuffer() });
try {
  const { createLinearAttentionRecovery, captureLinearAttentionRecovery, isLinearAttentionRecoveryRecording } = await import(moduleURL);
  for (const width of [0,1,NaN,1.5]) assert.throws(() => createLinearAttentionRecovery(recorder(),width), /bounds/);
  {
    const parent = recorder(), lease = createLinearAttentionRecovery(parent,4), layer = state();
    assert.equal(isLinearAttentionRecoveryRecording(parent),true);
    assert.throws(() => createLinearAttentionRecovery(parent,4), /verification window/);
    assert.throws(() => captureLinearAttentionRecovery(parent,layer,3,async()=>{}), /declared width/);
    assert.equal(parent.owned.size,0,'invalid width allocates nothing');
    let recovered = 0;
    captureLinearAttentionRecovery(parent,layer,4,async()=>{recovered++;});
    assert.equal(parent.owned.size,2);
    assert.throws(() => captureLinearAttentionRecovery(parent,layer,4,async()=>{}), /exactly once/);
    const other = { ...recorder(), device: {} };
    assert.throws(() => lease.recordRestore(other), /same-device/);
    await assert.rejects(() => lease.recordRecover(recorder(),{maxTokens:3}), /match/);
    await lease.recordRecover(parent,{maxTokens:4});
    assert.equal(recovered,1);
    parent.submitted = true;
    await assert.rejects(() => lease.recordRecover(parent,{maxTokens:4}), /same-device/);
    const settlement = recorder();
    lease.recordRestore(settlement);
    assert.equal(settlement.copies.length,2,'post-submission cancellation records only initial-state restoration');
    lease.close();lease.close();
    assert.equal(isLinearAttentionRecoveryRecording(parent),false);
    assert.equal(parent.owned.size,2,'closing a lease cannot recycle buffers referenced by unsubmitted recovery work');
    assert.throws(() => lease.recordRestore(settlement), /live parent lease/);
    settlement.abort();parent.abort();assert.equal(parent.owned.size,0);
  }
  {
    const parent = recorder(), lease = createLinearAttentionRecovery(parent,4);
    parent.submitted = true;
    assert.throws(() => captureLinearAttentionRecovery(parent,state(),4,async()=>{}), /unsubmitted window/);
    assert.equal(parent.owned.size,0,'a submitted parent cannot leak a new snapshot allocation');
    lease.close();
  }
  for (const failure of ['allocation','recording']) {
    allocations = 0;failAt = failure === 'allocation' ? 2 : Infinity;
    const parent = recorder(), lease = createLinearAttentionRecovery(parent,4);
    if (failure === 'allocation') assert.throws(() => captureLinearAttentionRecovery(parent,state(),4,async()=>{}), /allocation failure/);
    else {
      captureLinearAttentionRecovery(parent,state(),4,async()=>{throw Error('recovery recording failed');});
      await assert.rejects(() => lease.recordRecover(parent,{maxTokens:4}), /recording failed/);
    }
    assert.ok(parent.owned.size > 0,'partial resources remain with the command owner');
    lease.close();parent.abort();assert.equal(parent.owned.size,0);
  }
  console.log('speculative-recovery-ownership: passed');
} finally { hook.deregister();delete globalThis.recoveryOwnershipProbe; }
