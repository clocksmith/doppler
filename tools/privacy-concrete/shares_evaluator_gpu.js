/**
 * GPU evaluator for the two-server shares route. Same pipe protocol and manifest as shares_evaluator.py; the product
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

function argument(name) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) throw new Error(`Missing ${name}`);
  return process.argv[index + 1];
}

const directory = argument('--cache-dir');
const policy = JSON.parse(readFileSync(new URL('shares-policy.json', import.meta.url), 'utf8'));
const maximumRows = (1 + 2 * policy.checkRounds) * policy.maximumRows;
const manifest = JSON.parse(readFileSync(join(directory, 'shares-manifest.json'), 'utf8'));
if (manifest.schema !== 'doppler.two-server-shares-manifest/v1') throw new Error('Versioned shares manifest required');
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

function storage(bytes) {
  const size = Math.max(4, Math.ceil(bytes.byteLength / 4) * 4);
  if (size > device.limits.maxStorageBufferBindingSize) throw new Error('Matrix exceeds this GPU\'s storage buffer limit');
  const buffer = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
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
  return buffer;
}

/** Public coefficients for one projection: verified once, transposed to column order, offset to unsigned, kept on the GPU. */
const resident = new Map();
function matrix(name) {
  if (resident.has(name)) return resident.get(name);
  const row = projections.get(name);
  if (basename(row.file) !== row.file || row.dtype !== '<i2') throw new Error('Coefficient file must be a 16-bit file directly in the cache directory');
  const path = join(directory, row.file);
  const stat = lstatSync(path);
  if (!stat.isFile() || stat.size !== row.inputWidth * row.outputWidth * 2) throw new Error('Regular public coefficient file of the declared size required');
  const bytes = readFileSync(path);
  if (createHash('sha256').update(bytes).digest('hex') !== row.weightId) throw new Error('Public coefficient identity changed');
  const source = new Int16Array(bytes.buffer, bytes.byteOffset, row.inputWidth * row.outputWidth);
  const inputs = row.inputWidth, outputs = row.outputWidth;
  const transposed = new Uint16Array(outputs * inputs + (outputs * inputs & 1));
  for (let i = 0; i < inputs; i += 1) {
    const base = i * outputs;
    for (let j = 0; j < outputs; j += 1) transposed[j * inputs + i] = source[base + j] + 32768;
  }
  const entry = { inputs, outputs, buffer: storage(transposed) };
  resident.set(name, entry);
  return entry;
}

/** rows: BigUint64Array of count * inputs field elements (any 64-bit value is taken modulo the prime). */
async function product(name, rows, count) {
  const { inputs, outputs, buffer: weights } = matrix(name);
  const limbs = new Uint32Array(count * inputs * 4);
  const correction = new Uint32Array(count * 2);
  const words = new Uint32Array(rows.buffer, rows.byteOffset, count * inputs * 2);
  for (let r = 0; r < count; r += 1) {
    let sum = 0n;
    for (let i = 0; i < inputs; i += 1) {
      const at = r * inputs + i, lo = words[at * 2], hi = words[at * 2 + 1];
      limbs[at * 4] = lo & 0xffff; limbs[at * 4 + 1] = lo >>> 16; limbs[at * 4 + 2] = hi & 0xffff; limbs[at * 4 + 3] = hi >>> 16;
      sum += rows[at];
    }
    const offset = (sum % PRIME) * 32768n % PRIME;
    correction[r * 2] = Number(offset & 0xffffffffn); correction[r * 2 + 1] = Number(offset >> 32n);
  }
  const limbBuffer = storage(limbs), correctionBuffer = storage(correction);
  const resultBytes = count * outputs * 8;
  const results = device.createBuffer({ size: resultBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const staging = device.createBuffer({ size: resultBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const uniforms = [];
  let scoped = true;
  device.pushErrorScope('validation');
  try {
    const columnsPerDispatch = Math.max(WORKGROUP_COLUMNS,
      Math.floor(DISPATCH_BUDGET / (count * inputs) / WORKGROUP_COLUMNS) * WORKGROUP_COLUMNS);
    for (let begin = 0; begin < outputs; begin += columnsPerDispatch) {
      const columns = Math.min(columnsPerDispatch, outputs - begin);
      const uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      uniforms.push(uniform);
      device.queue.writeBuffer(uniform, 0, new Uint32Array([count, inputs, outputs, begin, columns, 0, 0, 0]));
      const bind = device.createBindGroup({ layout, entries: [
        { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: limbBuffer } },
        { binding: 2, resource: { buffer: weights } }, { binding: 3, resource: { buffer: correctionBuffer } },
        { binding: 4, resource: { buffer: results } }] });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(columns / WORKGROUP_COLUMNS), count, 1);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(results, 0, staging, 0, resultBytes);
    device.queue.submit([encoder.finish()]);
    // An error scope, not an uncaptured-error listener: the listener crashed the Node binding this was first run with.
    const failure = await device.popErrorScope();
    scoped = false;
    if (failure) throw new Error(`GPU validation: ${failure.message}`);
    await staging.mapAsync(GPUMapMode.READ);
    const answer = Buffer.from(new Uint8Array(staging.getMappedRange()));
    staging.unmap();
    return { answer, outputs };
  } finally {
    if (scoped) await device.popErrorScope();
    for (const buffer of [limbBuffer, correctionBuffer, results, staging, ...uniforms]) buffer.destroy();
  }
}

// Exact-length reads from stdin.
const chunks = [];
let buffered = 0, ended = false, wake = null;
process.stdin.on('data', chunk => { chunks.push(chunk); buffered += chunk.length; wake?.(); });
process.stdin.on('end', () => { ended = true; wake?.(); });
async function read(length) {
  while (buffered < length) {
    if (ended) throw new Error('end of stream');
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
  for (;;) {
    let magic;
    try { magic = (await read(4)).toString('latin1'); } catch { return; }
    if (magic === CLOSE) return;
    if (magic !== REQUEST) { respond(0n, 1, 0, 0); return; }
    const head = await read(REQUEST_HEAD_BYTES - 4);
    const sequence = head.readBigUInt64LE(0), nameBytes = head.readUInt16LE(8), rows = head.readUInt32LE(10), width = head.readUInt32LE(14);
    let name = null;
    if (nameBytes <= 1024) {
      try { name = new TextDecoder('utf-8', { fatal: true }).decode(await read(nameBytes)); } catch { name = null; }
    }
    const row = name === null ? undefined : projections.get(name);
    if (!row || width !== row.inputWidth || rows < 1 || rows > maximumRows) { respond(sequence, 1, 0, 0); return; }
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
}

await serve();
await new Promise(resolve => process.stdout.write('', resolve));
process.exit(0);
