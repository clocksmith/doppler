import assert from 'node:assert/strict';
import { validateQwen35RoutedConfig } from '../../src/config/schema/runtime/loading.schema.js';
import { RoutedExpertCache } from '../../src/loader/experts/routed-expert-cache.js';
import { planRoutedExpertBank } from '../../src/loader/experts/routed-expert-plan.js';

// CPU fault injection only. These checks do not measure physical GPU retirement.
globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 };

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

function fixture(direct) {
  const manifest = {
    architecture: { numLayers: 1, hiddenSize: 256 },
    moeConfig: { expertFormat: 'qwen3_5', numExperts: 4, expertIntermediateSize: 256 },
  };
  const locations = new Map();
  const shards = new Map();
  for (const [kind, shape] of [['gate_up', [4, 512, 256]], ['down', [4, 256, 256]]]) {
    const shardIndex = shards.size;
    const size = shape[0] * shape[1] * 144;
    const bytes = new Uint8Array(size);
    for (let expert = 0; expert < 4; expert++) {
      bytes.fill(expert + 1 + shardIndex * 16, expert * size / 4, (expert + 1) * size / 4);
    }
    shards.set(shardIndex, bytes);
    locations.set(`model.language_model.layers.0.mlp.experts.${kind}_proj`, {
      shape, size, dtype: 'Q4_K', layout: 'row', shardIndex, offset: 0,
    });
  }
  const plan = planRoutedExpertBank(manifest, locations, 0);
  const buffers = [];
  const reads = [];
  const destinations = new Set();
  const uploads = [];
  const hooks = {
    allocation() {},
    encode() {},
    finish() {},
    submit() {},
    async read() {},
    upload() {},
    async drain() {},
  };
  const gpu = {
    limits: { maxBufferSize: 16 * 1024 ** 2, maxStorageBufferBindingSize: 16 * 1024 ** 2 },
    createBuffer({ size, label, usage }) {
      hooks.allocation();
      const buffer = {
        size, label, usage, data: new Uint8Array(size), destroyed: false,
        destroy() { this.destroyed = true; },
      };
      buffers.push(buffer);
      return buffer;
    },
    createCommandEncoder() {
      const operations = [];
      return { copyBufferToBuffer(source, from, target, to, bytes) {
        hooks.encode(source, target);
        operations.push(() => target.data.set(source.data.subarray(from, from + bytes), to));
      }, finish() { hooks.finish(); return operations; } };
    },
    queue: {
      submit(commands) { hooks.submit(); for (const command of commands) for (const operation of command) operation(); },
      onSubmittedWorkDone: () => hooks.drain(),
      writeBuffer(buffer, offset, data) {
        assert.equal(buffer.destroyed, false, 'uploads never target retired buffers');
        hooks.upload(buffer, data);
        buffer.data.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
        uploads.push({ buffer, bytes: data.byteLength });
      },
    },
  };
  const policy = {
    enabled: true, storageMode: 'compact-decode', evictionPolicy: 'lru',
    compactLayerBytes: null, compactPrefillBytes: 0, compactPrefillReuse: 'none', prefillCompactAdmission: 'none', sourceReadMode: 'plane-serial', prefillUploadMode: 'expert-copy',
    cacheBytes: 2 * plan.expertBytes + plan.numExperts * 4,
    hostCacheBytes: 0, hostCacheAdmission: 'all', hostCacheAdmissionPhase: 'all',
    stagingBytes: 8192, readConcurrency: 2, routeChunkTokens: 2,
    gpuBudgetBytes: 16 * 1024 ** 2, reserveBytes: 1024 ** 2,
  };
  const options = {
    device: gpu, policy, compactLayerCount: 1, otherGpuBytes: () => 0,
    async readRange(shard, offset, length, signal) {
      reads.push({ shard, offset, length });
      await hooks.read(signal);
      return shards.get(shard).slice(offset, offset + length).buffer;
    },
    readRangeInto: direct ? async (shard, offset, destination, signal) => {
      destinations.add(destination.buffer);
      reads.push({ shard, offset, length: destination.byteLength });
      // A cancelled source is allowed to have partially written its destination.
      destination.fill(0xff);
      await hooks.read(signal);
      destination.set(shards.get(shard).subarray(offset, offset + destination.byteLength));
      return destination.byteLength;
    } : undefined,
  };
  return { cache: new RoutedExpertCache(options), options, plan, buffers, reads, destinations, uploads, hooks, policy, shards };
}

function checkWeights(f, weights, experts) {
  const slots = new Uint32Array(weights.sourceExpertToSlot.data.buffer);
  for (const plane of f.plan.planes) {
    for (const expert of experts) {
      assert(slots[expert] < weights.numExperts);
      const offset = slots[expert] * plane.expertBytes;
      assert.deepEqual(weights[plane.role].buffer.data.slice(offset, offset + plane.expertBytes),
        f.shards.get(plane.location.shardIndex).slice(expert * plane.expertBytes, (expert + 1) * plane.expertBytes));
    }
  }
}


function enable(f) {
  f.policy.compactPrefillReuse = 'resident-gpu';
  f.policy.compactPrefillBytes = 1024 * 1024;
  f.policy.cacheBytes += f.policy.compactPrefillBytes;
}
function fullBytes(f, weights, experts) {
  for (const plane of f.plan.planes) for (const expert of experts) {
    assert.deepEqual(weights[plane.role].buffer.data.slice(expert*plane.expertBytes,(expert+1)*plane.expertBytes),
      f.shards.get(plane.location.shardIndex).slice(expert*plane.expertBytes,(expert+1)*plane.expertBytes));
  }
}
async function seeded() {
  const f = fixture(true); enable(f);
  checkWeights(f, await f.cache.prepare(f.plan,[2,3],undefined,'decode'),[2,3]);
  return f;
}
{
  const f=await seeded(), before=f.reads.length;
  fullBytes(f,await f.cache.prepare(f.plan,[2,3,2],undefined,'prefill'),[2,3]);
  assert.equal(f.reads.length,before,'resident experts need no source reread');
  assert.equal(f.cache.getStats().prefillGpuReuseExperts,2);
  const copied=f.cache.getStats().prefillGpuReuseExperts;
  fullBytes(f,await f.cache.prepare(f.plan,[0,2,3],undefined,'prefill'),[0,2,3]);
  assert.equal(f.cache.getStats().prefillGpuReuseExperts,copied,'already loaded experts are not copied twice');
  assert(f.reads.length>before,'missing expert is authenticated through ordinary source path');
  const reads=f.reads.length;
  checkWeights(f,await f.cache.prepare(f.plan,[2,3],undefined,'decode'),[2,3]);
  assert.equal(f.reads.length,reads,'copyback must not change decode retention');
  await f.cache.destroy();assert.equal(f.cache.getStats().residentBytes,0);
}
for(const failure of ['encode','finish','abort-before-submit','submit']) {
  const f=await seeded(), controller=new AbortController();
  const inject=()=>{throw new Error('Injected '+failure)};
  if(failure==='abort-before-submit')f.hooks.finish=()=>controller.abort(new Error('Injected abort'));
  else f.hooks[failure]=inject;
  await assert.rejects(f.cache.prepare(f.plan,[2,3],controller.signal,'prefill'));
  f.hooks.encode=()=>{}; f.hooks.finish=()=>{}; f.hooks.submit=()=>{};
  const before=f.reads.length;
  fullBytes(f,await f.cache.prepare(f.plan,[2,3],undefined,'prefill'),[2,3]);
  assert.equal(f.reads.length,before,'retry retains independently owned source slots');
  await f.cache.destroy();assert(f.buffers.every(b=>b.destroyed));
}
{
  const f=await seeded(), started=deferred(), drain=deferred();let fail=false,settled=false;
  f.hooks.read=async()=>{fail=true;throw new Error('Injected read after resident copy')};
  f.hooks.drain=async()=>{if(fail){started.resolve();await drain.promise}};
  const pending=f.cache.prepare(f.plan,[2,0],undefined,'prefill').then(()=>assert.fail('accepted failed read'),error=>{settled=true;return error});
  await started.promise;assert.equal(settled,false,'failed preparation keeps ownership until copied destinations settle');
  const closing=f.cache.destroy();await Promise.resolve();assert.equal(settled,false);
  drain.resolve();await pending;await closing;
  assert(f.buffers.every(b=>b.destroyed));assert.equal(f.cache.getStats().residentBytes,0);
}
{
  const a=await seeded(),b=await seeded();await a.cache.destroy();
  fullBytes(b,await b.cache.prepare(b.plan,[2,3],undefined,'prefill'),[2,3]);
  assert(b.buffers.some(x=>!x.destroyed));await b.cache.destroy();
}
console.log('resident compact-to-prefill copies: exact planes, misses, retention, fault/drain/retry and independent owners passed (CPU mocks)');

{
  const f = fixture(true);
  assert.throws(() => validateQwen35RoutedConfig({ ...f.policy, compactPrefillReuse: 'unknown' }), /compactPrefillReuse/);
  assert.throws(() => validateQwen35RoutedConfig({ ...f.policy, compactPrefillReuse: 'resident-gpu' }), /reservation/);
  const { compactPrefillReuse, ...missing } = f.policy;
  assert.throws(() => validateQwen35RoutedConfig(missing), /compactPrefillReuse/);
  await f.cache.destroy();
}

// Copying an older slot for prefill must not promote it in decode LRU order.
{
  const f = await seeded();
  await f.cache.prepare(f.plan, [2], undefined, 'prefill');
  await f.cache.prepare(f.plan, [0], undefined, 'decode');
  const before = f.reads.length;
  checkWeights(f, await f.cache.prepare(f.plan, [3], undefined, 'decode'), [3]);
  assert.equal(f.reads.length, before, 'newer decode slot must survive prefill reuse');
  checkWeights(f, await f.cache.prepare(f.plan, [2], undefined, 'decode'), [2]);
  assert(f.reads.length > before, 'older reused slot retains its eviction order');
  await f.cache.destroy();
}
