import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const target = new URL('../../src/storage/shard-manager.js', import.meta.url).href;
let fullReads = 0;
const backend = { init: async () => {}, openModel: async () => null, cleanup: async () => {},
  readFile: async () => { fullReads++; return new ArrayBuffer(64); } };
globalThis.boundedRangeBackend = backend;
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL !== target) return next(specifier, context);
  let source;
  if (specifier === './backend-selection.js') source = 'export const selectStorageBackend = () => ({ type: "probe", backend: globalThis.boundedRangeBackend });';
  if (specifier === '../formats/rdrr/index.js') source = `
    export const getManifest = () => ({ shards: [{ filename: 'checkpoint.bin', size: 64 }] });
    export const getShardInfo = () => ({ filename: 'checkpoint.bin', size: 64 });
    export const getShardCount = () => 1;
    export const getExpectedShardHash = () => null;
    export const generateShardFilename = () => 'checkpoint.bin';`;
  return source ? { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true } : next(specifier, context);
} });
try {
  const { openModelStore, loadShardRange, cleanup } = await import(target);
  await openModelStore('bounded-table');
  await assert.rejects(loadShardRange(0, 4, 4, { requireRange: true }), /bounded backend range reader/);
  assert.equal(fullReads, 0);
  backend.readFileRange = async (_name, offset, length) => Uint8Array.from({ length }, (_, i) => offset + i).buffer;
  assert.deepEqual([...new Uint8Array(await loadShardRange(0, 4, 4, { requireRange: true }))], [4, 5, 6, 7]);
  await assert.rejects(loadShardRange(0, 4, 4, { requireRange: true, verify: true }), /verification is disabled/);
  await assert.rejects(loadShardRange(0, 4, 4, { requireRange: true, verify: true, tensorId: 'embedding' }), /verification is disabled/);
  assert.equal(fullReads, 0);
  await cleanup();
} finally { hooks.deregister(); delete globalThis.boundedRangeBackend; }
console.log('bounded shard range: required range reads cannot materialize a full file or shard');
