/**
 * GPU evaluator for the two-server shares route. Same pipe protocol and manifest as the retired CPU evaluator (commit 3e485ca93); the product
 * runs in shares_field_matmul.wgsl through a Node WebGPU binding. It multiplies whatever rows it is sent by the named
 * public matrix modulo 2^61 - 1. The rows are uniformly random numbers, so it handles no private data and needs no key.
 *
 * Usage: node shares_evaluator_gpu.js --cache-dir DIR --webgpu-module /path/to/webgpu/index.js
 */
import { createHash } from 'node:crypto';
import { lstatSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { basename, join } from 'node:path';

const PRIME = (1n << 61n) - 1n;
const REQUEST = 'SHQ1', RESPONSE = 'SHR1', CLOSE = 'SHC1';
const REQUEST_HEAD_BYTES = 22, RESPONSE_HEAD_BYTES = 21;
const WORKGROUP_COLUMNS = 64;
// Inner-loop iterations per dispatch, so one command buffer stays short on any GPU.
const DISPATCH_BUDGET = 400_000_000;
const MAX_INPUT_WIDTH = 1 << 21;
const SUPPORTED_DTYPES = new Set(['<i2', 'int16', '|i1', '<i1', 'int8']);

class StreamEndedError extends Error {}
process.stdout.on('error', () => {});

function argument(name) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) throw new Error(`Missing ${name}`);
  return process.argv[index + 1];
}

function validProjection(row) {
  return Boolean(
    row
    && typeof row.name === 'string'
    && typeof row.file === 'string'
    && basename(row.file) === row.file
    && SUPPORTED_DTYPES.has(row.dtype)
    && typeof row.weightId === 'string'
    && Number.isInteger(row.inputWidth)
    && row.inputWidth >= 1
    && row.inputWidth < MAX_INPUT_WIDTH
    && Number.isInteger(row.outputWidth)
    && row.outputWidth >= 1
    && row.inputWidth * row.outputWidth <= 0xffffffff
  );
}

const directory = argument('--cache-dir');
const policy = JSON.parse(readFileSync(new URL('shares-policy.json', import.meta.url), 'utf8'));
const maximumRows = (1 + 2 * policy.checkRounds) * policy.maximumRows;
const manifest = JSON.parse(readFileSync(join(directory, 'shares-manifest.json'), 'utf8'));
if (manifest.schema !== 'doppler.two-server-shares-manifest/v1' || !Array.isArray(manifest.projections)) {
  throw new Error('Versioned shares manifest required');
}
const projections = new Map(manifest.projections.map(row => [row.name, row]));

const webgpu = createRequire(import.meta.url)(argument('--webgpu-module'));
Object.assign(globalThis, webgpu.globals);
// Keep the instance referenced for the life of the process: if it is collected, the device dies with it.
const instance = webgpu.create([]);
globalThis.sharesGpuInstance = instance;
const adapter = await instance.requestAdapter();
if (!adapter) throw new Error('No WebGPU adapter');
const device = await adapter.requestDevice({ requiredLimits: {
  maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
// The synchronous form: the asynchronous one crashes the Node binding this was first run with.
const pipeline = device.createComputePipeline({ layout: 'auto',
  compute: { module: device.createShaderModule({ code: readFileSync(new URL('shares_field_matmul.wgsl', import.meta.url), 'utf8') }), entryPoint: 'main' } });
const layout = pipeline.getBindGroupLayout(0);
const maxStorageLimit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
const maxColumnsPerDispatch = device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_COLUMNS;

function uploadBytes(buffer, bytes) {
  const view = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const whole = Math.floor(view.length / 4) * 4;
  // Upload in bounded pieces; a trailing partial word is padded with zeros.
  for (let offset = 0; offset < whole; offset += 64 * 1024 * 1024) {
    device.queue.writeBuffer(buffer, offset, view.subarray(offset, offset + Math.min(64 * 1024 * 1024, whole - offset)));
  }
  if (whole < view.length) {
    const tail = new Uint8Array(4); tail.set(view.subarray(whole));
    device.queue.writeBuffer(buffer, whole, tail);
  }
}

function storage(bytes) {
  const size = Math.max(4, Math.ceil(bytes.byteLength / 4) * 4);
  if (size > maxStorageLimit) throw new Error('Matrix exceeds this GPU\'s storage buffer limit');
  const buffer = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  uploadBytes(buffer, bytes);
  return buffer;
}

function chooseCapacity(current, needed, limit) {
  const aligned = Math.max(8, Math.ceil(needed / 8) * 8);
  if (aligned > limit) throw new Error('Projection exceeds this GPU\'s buffer limit');
  if (current >= aligned) return current;
  let next = Math.max(256, current);
  while (next < aligned && next <= limit / 2) next *= 2;
  return next >= aligned ? next : aligned;
}

/** Reusable GPU and host scratch buffers across calls; grown monotonically to high-water mark and destroyed on exit. */
const workspace = {
  generation: 1,
  limbs: null, limbsCapacity: 0,
  correction: null, correctionCapacity: 0,
  results: null, resultsCapacity: 0,
  staging: null, stagingCapacity: 0,
  uniforms: [],
  hostLimbs: new Uint32Array(0),
  hostCorrection: new Uint32Array(0),
  uniformScratch: new Uint32Array(8),
};

function ensureGpuBuffer(bufferKey, capacityKey, needed, usage, limit, invalidatesBindGroups) {
  const next = chooseCapacity(workspace[capacityKey], needed, limit);
  if (workspace[bufferKey] && workspace[capacityKey] >= next) return workspace[bufferKey];
  workspace[bufferKey]?.destroy();
  workspace[bufferKey] = device.createBuffer({ size: next, usage });
  workspace[capacityKey] = next;
  if (invalidatesBindGroups) workspace.generation += 1;
  return workspace[bufferKey];
}

function ensureHostWords(key, neededWords) {
  if (workspace[key].length >= neededWords) return workspace[key];
  let next = Math.max(256, workspace[key].length);
  while (next < neededWords) next *= 2;
  workspace[key] = new Uint32Array(next);
  return workspace[key];
}

function uniformBuffer(chunkIndex) {
  while (workspace.uniforms.length <= chunkIndex) {
    workspace.uniforms.push(device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  }
  return workspace.uniforms[chunkIndex];
}

/** Public coefficients for one projection: verified once, transposed to column order, offset to unsigned, kept on the GPU. */
const resident = new Map();
function matrix(name) {
  if (resident.has(name)) return resident.get(name);
  const row = projections.get(name);
  if (!validProjection(row)) throw new Error('Coefficient file must sit directly in the cache directory with admitted geometry');
  const itemBytes = (row.dtype === '<i2' || row.dtype === 'int16') ? 2 : 1;
  const path = join(directory, row.file);
  const stat = lstatSync(path);
  if (!stat.isFile() || stat.size !== row.inputWidth * row.outputWidth * itemBytes) {
    throw new Error('Regular public coefficient file of the declared size required');
  }
  const bytes = readFileSync(path);
  if (createHash('sha256').update(bytes).digest('hex') !== row.weightId) throw new Error('Public coefficient identity changed');
  const elementCount = row.inputWidth * row.outputWidth;
  let source;
  if (itemBytes === 2) {
    const aligned = (bytes.byteOffset & 1) === 0 ? bytes : Buffer.from(bytes);
    source = new Int16Array(aligned.buffer, aligned.byteOffset, elementCount);
  } else {
    source = new Int8Array(bytes.buffer, bytes.byteOffset, elementCount);
  }
  const inputs = row.inputWidth, outputs = row.outputWidth;
  const transposed = new Uint16Array(outputs * inputs + (outputs * inputs & 1));
  for (let i = 0; i < inputs; i += 1) {
    const base = i * outputs;
    for (let j = 0; j < outputs; j += 1) transposed[j * inputs + i] = source[base + j] + 32768;
  }
  const entry = { inputs, outputs, buffer: storage(transposed), bindGeneration: 0, bindGroups: [] };
  resident.set(name, entry);
  return entry;
}

function bindGroupFor(entry, chunkIndex, limbBuffer, correctionBuffer, resultsBuffer) {
  if (entry.bindGeneration !== workspace.generation) {
    entry.bindGeneration = workspace.generation;
    entry.bindGroups.length = 0;
  }
  if (!entry.bindGroups[chunkIndex]) {
    entry.bindGroups[chunkIndex] = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer(chunkIndex) } },
        { binding: 1, resource: { buffer: limbBuffer } },
        { binding: 2, resource: { buffer: entry.buffer } },
        { binding: 3, resource: { buffer: correctionBuffer } },
        { binding: 4, resource: { buffer: resultsBuffer } },
      ],
    });
  }
  return entry.bindGroups[chunkIndex];
}

/** rows: BigUint64Array of count * inputs field elements (any 64-bit value is taken modulo the prime). */
async function product(name, rows, count) {
  const entry = matrix(name);
  const { inputs, outputs } = entry;
  if (count > device.limits.maxComputeWorkgroupsPerDimension || count * inputs > 0x3fffffff || count * outputs > 0x7fffffff) {
    throw new Error('Batch exceeds 32-bit GPU indexing bounds');
  }
  const limbWords = count * inputs * 4;
  const correctionWords = count * 2;
  const limbs = ensureHostWords('hostLimbs', limbWords).subarray(0, limbWords);
  const correction = ensureHostWords('hostCorrection', correctionWords).subarray(0, correctionWords);
  const words = new Uint32Array(rows.buffer, rows.byteOffset, count * inputs * 2);
  for (let r = 0; r < count; r += 1) {
    // inputs < 2^21 and each 32-bit half < 2^32, so sumLo and sumHi stay strictly below 2^53 - 1 (exact in IEEE-754 double).
    let sumLo = 0, sumHi = 0;
    const rowBase = r * inputs;
    for (let i = 0; i < inputs; i += 1) {
      const at = rowBase + i, wordAt = at * 2, limbAt = at * 4;
      const lo = words[wordAt], hi = words[wordAt + 1];
      limbs[limbAt] = lo & 0xffff; limbs[limbAt + 1] = lo >>> 16; limbs[limbAt + 2] = hi & 0xffff; limbs[limbAt + 3] = hi >>> 16;
      sumLo += lo; sumHi += hi;
    }
    const rowSum = BigInt(sumLo) + (BigInt(sumHi) << 32n);
    const offset = (rowSum % PRIME) * 32768n % PRIME;
    correction[r * 2] = Number(offset & 0xffffffffn); correction[r * 2 + 1] = Number(offset >> 32n);
  }
  const resultBytes = count * outputs * 8;
  let scoped = true, mapped = false;
  device.pushErrorScope('validation');
  let staging = null;
  try {
    const limbBuffer = ensureGpuBuffer('limbs', 'limbsCapacity', limbWords * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, maxStorageLimit, true);
    const correctionBuffer = ensureGpuBuffer('correction', 'correctionCapacity', correctionWords * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, maxStorageLimit, true);
    const results = ensureGpuBuffer('results', 'resultsCapacity', resultBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, maxStorageLimit, true);
    staging = ensureGpuBuffer('staging', 'stagingCapacity', resultBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, device.limits.maxBufferSize, false);
    uploadBytes(limbBuffer, limbs);
    uploadBytes(correctionBuffer, correction);

    const columnsPerDispatch = Math.min(
      maxColumnsPerDispatch,
      Math.max(WORKGROUP_COLUMNS, Math.floor(DISPATCH_BUDGET / (count * inputs) / WORKGROUP_COLUMNS) * WORKGROUP_COLUMNS),
    );
    const uniformScratch = workspace.uniformScratch;
    uniformScratch[0] = count; uniformScratch[1] = inputs; uniformScratch[2] = outputs;
    for (let begin = 0, chunkIndex = 0; begin < outputs; begin += columnsPerDispatch, chunkIndex += 1) {
      const columns = Math.min(columnsPerDispatch, outputs - begin);
      const isLastChunk = begin + columns >= outputs;
      const uniform = uniformBuffer(chunkIndex);
      uniformScratch[3] = begin; uniformScratch[4] = columns;
      device.queue.writeBuffer(uniform, 0, uniformScratch);
      const bind = bindGroupFor(entry, chunkIndex, limbBuffer, correctionBuffer, results);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(columns / WORKGROUP_COLUMNS), count, 1);
      pass.end();
      if (isLastChunk) encoder.copyBufferToBuffer(results, 0, staging, 0, resultBytes);
      device.queue.submit([encoder.finish()]);
    }
    // An error scope, not an uncaptured-error listener: the listener crashed the Node binding this was first run with.
    const errorPromise = device.popErrorScope();
    scoped = false;
    await staging.mapAsync(GPUMapMode.READ, 0, resultBytes);
    mapped = true;
    const failure = await errorPromise;
    if (failure) throw new Error(`GPU validation: ${failure.message}`);
    const answer = Buffer.allocUnsafe(resultBytes);
    answer.set(new Uint8Array(staging.getMappedRange(0, resultBytes)));
    staging.unmap();
    mapped = false;
    return { answer, outputs };
  } finally {
    if (mapped && staging) staging.unmap();
    if (scoped) await device.popErrorScope();
  }
}

function destroyGpuResources() {
  for (const entry of resident.values()) entry.buffer.destroy();
  resident.clear();
  for (const buffer of [workspace.limbs, workspace.correction, workspace.results, workspace.staging, ...workspace.uniforms]) {
    buffer?.destroy();
  }
  workspace.limbs = workspace.correction = workspace.results = workspace.staging = null;
  workspace.uniforms.length = 0;
}

// Exact-length reads from stdin.
const chunks = [];
let buffered = 0, ended = false, wake = null;
process.stdin.on('data', chunk => { chunks.push(chunk); buffered += chunk.length; wake?.(); });
process.stdin.on('end', () => { ended = true; wake?.(); });
async function read(length) {
  while (buffered < length) {
    if (ended) throw new StreamEndedError('end of stream');
    await new Promise(resolve => { wake = resolve; });
    wake = null;
  }
  const all = chunks.length === 1 ? chunks[0] : Buffer.concat(chunks);
  chunks.length = 0;
  const out = all.subarray(0, length);
  if (all.length > length) chunks.push(all.subarray(length));
  buffered -= length;
  return out;
}

function respond(sequence, status, rows, width, payload) {
  const head = Buffer.alloc(RESPONSE_HEAD_BYTES);
  head.write(RESPONSE, 0, 'latin1'); head.writeBigUInt64LE(sequence, 4); head.writeUInt8(status, 12);
  head.writeUInt32LE(rows, 13); head.writeUInt32LE(width, 17);
  process.stdout.write(head);
  if (payload) process.stdout.write(payload);
}

async function serve() {
  try {
    for (;;) {
      const magic = (await read(4)).toString('latin1');
      if (magic === CLOSE) return;
      if (magic !== REQUEST) { respond(0n, 1, 0, 0); return; }
      const head = await read(REQUEST_HEAD_BYTES - 4);
      const sequence = head.readBigUInt64LE(0), nameBytes = head.readUInt16LE(8), rows = head.readUInt32LE(10), width = head.readUInt32LE(14);
      let name = null;
      if (nameBytes <= 1024) {
        const nameBuf = await read(nameBytes);
        try { name = new TextDecoder('utf-8', { fatal: true }).decode(nameBuf); } catch { name = null; }
      }
      const row = name === null ? undefined : projections.get(name);
      if (!validProjection(row) || width !== row.inputWidth || rows < 1 || rows > maximumRows) { respond(sequence, 1, 0, 0); return; }
      const payload = await read(8 * rows * width);
      const aligned = new BigUint64Array(rows * width);
      new Uint8Array(aligned.buffer).set(payload);
      try {
        const { answer, outputs } = await product(name, aligned, rows);
        respond(sequence, 0, rows, outputs, answer);
      } catch (error) {
        process.stderr.write(`shares evaluator: ${String(error.message).slice(0, 400)}\n`);
        respond(sequence, 1, 0, 0);
        return;
      }
    }
  } catch (error) {
    if (!(error instanceof StreamEndedError)) throw error;
  }
}

try {
  await serve();
  await new Promise(resolve => process.stdout.write('', resolve));
} finally {
  destroyGpuResources();
}
process.exit(0);
