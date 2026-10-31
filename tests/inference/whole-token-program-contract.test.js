import assert from 'node:assert/strict';

// Whole-token program contract on a fake device: the token input and the valid
// positions come from operation declarations, not from which observed words differ.

globalThis.GPUBufferUsage ??= {
  MAP_READ: 0x1, MAP_WRITE: 0x2, COPY_SRC: 0x4, COPY_DST: 0x8, INDEX: 0x10,
  VERTEX: 0x20, UNIFORM: 0x40, STORAGE: 0x80, INDIRECT: 0x100, QUERY_RESOLVE: 0x200,
};
globalThis.GPUMapMode ??= { READ: 0x1, WRITE: 0x2 };

const { WholeTokenProgram } = await import('../../src/inference/pipelines/text/generator/whole-token-program.js');
const { declareStepContract } = await import('../../src/gpu/step-contract.js');

const U = GPUBufferUsage;

function makeBuffer(label, size, usage) {
  const bytes = new Uint8Array(size);
  return {
    label, size, usage, bytes,
    async mapAsync() {},
    getMappedRange(offset = 0, length = size - offset) { return bytes.slice(offset, offset + length).buffer; },
    unmap() {},
    destroy() {},
  };
}

function makeDevice() {
  const device = {
    dropNextSubmit: false,
    limits: { minUniformBufferOffsetAlignment: 256 },
    createBuffer: ({ label, size, usage }) => makeBuffer(label, size, usage),
    createBindGroup: (desc) => ({ label: desc.label, entries: desc.entries }),
    createCommandEncoder() {
      const ops = [];
      let pipeline = null;
      const groups = [];
      return {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          ops.push(() => dst.bytes.set(src.bytes.subarray(srcOffset, srcOffset + size), dstOffset));
        },
        clearBuffer(buffer, offset = 0, size = buffer.size - offset) { ops.push(() => buffer.bytes.fill(0, offset, offset + size)); },
        beginComputePass: () => ({
          setPipeline(p) { pipeline = p; },
          setBindGroup(index, group) { groups[index] = group; },
          dispatchWorkgroups() { const p = pipeline; const g = [...groups]; ops.push(() => p.run(g[0])); },
          end() {},
        }),
        finish: () => ({ ops }),
      };
    },
    queue: {
      writeBuffer(buffer, offset, data, dataOffset = 0, size) {
        const elem = ArrayBuffer.isView(data) ? (data.BYTES_PER_ELEMENT ?? 1) : 1;
        const base = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
        const start = dataOffset * elem;
        const length = size === undefined ? base.byteLength - start : size * elem;
        buffer.bytes.set(base.subarray(start, start + length), offset);
      },
      submit(commandBuffers) {
        if (device.dropNextSubmit) { device.dropNextSubmit = false; return; }
        for (const commandBuffer of commandBuffers) for (const op of commandBuffer.ops) op();
      },
    },
  };
  return device;
}

function makeRecorder(device) {
  let encoder = null;
  const raw = () => (encoder ??= device.createCommandEncoder());
  return {
    device,
    getEncoder: () => raw(),
    beginComputePass: () => raw().beginComputePass(),
    recordDispatch(pipeline, bindGroup, workgroups) {
      const pass = raw().beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(...workgroups);
      pass.end();
    },
    recordDispatchIndirect() { throw new Error('unused'); },
    createTempBuffer: (size, usage, label) => device.createBuffer({ size, usage, label }),
    trackTemporaryBuffer() {},
    submitPartial() {},
    submit() { device.queue.submit([raw().finish()]); encoder = null; },
  };
}

const word = (buffer, offset = 0) => new DataView(buffer.bytes.buffer).getUint32(offset, true);

// One fake decode step: out = token + 1000 * pos, appended to a KV row and read back.
function makeModel(device, { kvRows = 64 } = {}) {
  const res = {
    tokens: device.createBuffer({ label: 'tokens', size: 4, usage: U.STORAGE | U.COPY_DST }),
    uniform: device.createBuffer({ label: 'step_uniform', size: 16, usage: U.UNIFORM | U.COPY_DST }),
    out: device.createBuffer({ label: 'out', size: 4, usage: U.STORAGE | U.COPY_SRC }),
    kv: device.createBuffer({ label: 'kv', size: kvRows * 4, usage: U.STORAGE | U.COPY_DST }),
    staging: device.createBuffer({ label: 'staging', size: 4, usage: U.MAP_READ | U.COPY_DST }),
    layout: { label: 'layout' },
    pipeline: {
      run(group) {
        const entry = (binding) => group.entries.find((e) => e.binding === binding).resource;
        const u = entry(0);
        const t = entry(1);
        const pos = word(u.buffer, u.offset ?? 0);
        const token = word(t.buffer, t.offset ?? 0);
        new DataView(entry(2).buffer.bytes.buffer).setUint32(0, token + 1000 * pos, true);
      },
    },
  };
  return res;
}

function recordStep(device, recorder, res, pos, token, options) {
  const {
    declareToken = true, kvMax = Infinity, regime = null, uniformBytes = 16, writeToken = true, stepBase = null,
  } = options;
  if (declareToken) declareStepContract(recorder, { kind: 'token-input', buffer: res.tokens, offset: 0, count: 1 });
  if (writeToken) device.queue.writeBuffer(res.tokens, 0, new Uint32Array([token]));
  declareStepContract(recorder, { kind: 'position-range', owner: 'kv-cache', min: 0, max: kvMax });
  if (regime) declareStepContract(recorder, { kind: 'position-range', owner: 'regime', ...regime(pos) });
  // Word 1 counts decode steps from stepBase when given: affine in the position inside
  // one generation, but on a different line in a generation that starts elsewhere.
  device.queue.writeBuffer(res.uniform, 0, new Uint32Array([pos, stepBase === null ? 0 : pos - stepBase, 0, 0])
    .subarray(0, uniformBytes / 4));
  const group = device.createBindGroup({
    label: 'step', layout: res.layout, entries: [
      { binding: 0, resource: { buffer: res.uniform } },
      { binding: 1, resource: { buffer: res.tokens } },
      { binding: 2, resource: { buffer: res.out } },
    ],
  });
  recorder.recordDispatch(res.pipeline, group, [1]);
  recorder.getEncoder().copyBufferToBuffer(res.out, 0, res.kv, pos * 4, 4);
  recorder.getEncoder().copyBufferToBuffer(res.out, 0, res.staging, 0, 4);
  recorder.submit();
}

async function observeStep(program, device, res, pos, token, options = {}, method = 'observe') {
  const identity = options.identity ?? ['plan-a'];
  return program[method](pos, token, async (attach) => {
    const recorder = makeRecorder(device);
    attach(recorder);
    recordStep(device, recorder, res, pos, token, options);
    return word(res.staging);
  }, () => identity);
}

// The traced failure: four identical input tokens make the token write constant.
// The declared token input still feeds each replayed token.
{
  const device = makeDevice();
  const res = makeModel(device);
  const program = new WholeTokenProgram(device);
  for (const pos of [10, 11, 12, 13]) assert.equal(await observeStep(program, device, res, pos, 42), 42 + 1000 * pos);
  assert.equal(program.ready, true);
  assert.equal(program.rejects(14, ['plan-a']), null);
  assert.equal((await program.run(14, 99))[0], 99 + 14000);
  assert.equal((await program.run(15, 5))[0], 5 + 15000);
  assert.equal(word(res.kv, 15 * 4), 5 + 15000);
  program.destroy();
}

// Without a declared token input, identical tokens cannot be told from a constant: fail closed.
{
  const device = makeDevice();
  const res = makeModel(device);
  const program = new WholeTokenProgram(device);
  for (const pos of [10, 11, 12]) await observeStep(program, device, res, pos, 42, { declareToken: false });
  await assert.rejects(observeStep(program, device, res, 13, 42, { declareToken: false }), /No operation declared the decode step's input token/);
  program.destroy();
}

// An undeclared word that tracks the token is rejected rather than inferred.
{
  const device = makeDevice();
  const res = makeModel(device);
  const program = new WholeTokenProgram(device);
  const tokens = [7, 1234, 99, 5];
  for (let i = 0; i < 3; i++) await observeStep(program, device, res, 10 + i, tokens[i], { declareToken: false });
  await assert.rejects(observeStep(program, device, res, 13, tokens[3], { declareToken: false }), /tracks the input token/);
  program.destroy();
}

// A declared token input that is never written fails the step.
{
  const device = makeDevice();
  const res = makeModel(device);
  const program = new WholeTokenProgram(device);
  await assert.rejects(observeStep(program, device, res, 10, 42, { writeToken: false }), /declared .* but never written/);
  program.destroy();
}

// Declared ranges and copy bounds limit the replay positions; a rejected step
// invalidates the program and the next steps are observed again.
{
  const device = makeDevice();
  const res = makeModel(device);
  const program = new WholeTokenProgram(device);
  for (const pos of [10, 11, 12, 13]) await observeStep(program, device, res, pos, 42, { kvMax: 15 });
  assert.deepEqual(program.positionRange, { owner: 'kv-cache', min: 0, max: 15 });
  assert.equal(program.rejects(14, ['plan-a']), null);
  assert.match(program.rejects(14, ['plan-b']), /execution plan or a bound resource changed/);
  assert.match(program.rejects(15, ['plan-a']), /not the next position 14/);
  await program.run(14, 1);
  await program.run(15, 2);
  assert.match(program.rejects(16, ['plan-a']), /outside \[0, 15\] \(bounded by kv-cache\)/);
  program.invalidate('test');
  assert.equal(program.ready, false);
  assert.equal(program.stats.invalidations, 1);
  for (const pos of [16, 17, 18, 19]) await observeStep(program, device, res, pos, 3, { kvMax: 40 });
  assert.equal(program.ready, true);
  assert.equal((await program.run(20, 8))[0], 8 + 20000);
  program.destroy();
}
{
  const device = makeDevice();
  const res = makeModel(device, { kvRows: 32 });
  const program = new WholeTokenProgram(device);
  for (const pos of [10, 11, 12, 13]) await observeStep(program, device, res, pos, 42);
  assert.deepEqual(program.positionRange, { owner: 'copy to "kv"', min: 0, max: 31 });
  program.destroy();
}

// A declared threshold inside the observation window restarts observation after it;
// so does a change of the bound resources.
{
  const device = makeDevice();
  const res = makeModel(device);
  const program = new WholeTokenProgram(device);
  const regime = (pos) => (pos <= 11 ? { min: 0, max: 11 } : { min: 12, max: Infinity });
  for (const pos of [10, 11, 12, 13, 14]) await observeStep(program, device, res, pos, 42, { regime });
  assert.equal(program.ready, false);
  await observeStep(program, device, res, 15, 42, { regime });
  assert.equal(program.ready, true);
  assert.equal(program.stats.restarts, 1);
  assert.equal(program.positionRange.min, 12);
  program.destroy();
}
{
  const device = makeDevice();
  const res = makeModel(device);
  const program = new WholeTokenProgram(device);
  await observeStep(program, device, res, 10, 42, { identity: ['plan-a'] });
  await observeStep(program, device, res, 11, 42, { identity: ['plan-a'] });
  for (const pos of [12, 13, 14]) await observeStep(program, device, res, pos, 42, { identity: ['plan-b'] });
  assert.equal(program.ready, false);
  await observeStep(program, device, res, 15, 42, { identity: ['plan-b'] });
  assert.equal(program.ready, true);
  assert.equal(program.rejects(16, ['plan-a']), 'the execution plan or a bound resource changed');
  program.destroy();
}

// A binding that reads uniform bytes the step never wrote cannot be replayed.
{
  const device = makeDevice();
  const res = makeModel(device);
  const program = new WholeTokenProgram(device);
  await assert.rejects(observeStep(program, device, res, 10, 42, { uniformBytes: 4 }), /reads bytes the step did not write/);
  program.destroy();
}

// A replay whose command buffer never ran is an error, not the previous token.
{
  const device = makeDevice();
  const res = makeModel(device);
  const program = new WholeTokenProgram(device);
  for (const pos of [10, 11, 12, 13]) await observeStep(program, device, res, pos, 42);
  device.dropNextSubmit = true;
  await assert.rejects(program.run(14, 99), /wrote no sampled token/);
  program.destroy();
}

// A kept program replays a later generation after its first observed step matches it,
// at a position far from the steps it was built from.
{
  const device = makeDevice();
  const res = makeModel(device);
  const program = new WholeTokenProgram(device);
  for (const pos of [10, 11, 12, 13]) await observeStep(program, device, res, pos, 42);
  await program.run(14, 1);
  program.beginGeneration();
  assert.equal(program.awaitingVerification, true);
  assert.equal(await observeStep(program, device, res, 40, 7, {}, 'verify'), 7 + 40000);
  assert.equal(program.ready, true);
  assert.equal(program.awaitingVerification, false);
  assert.deepEqual([program.stats.reused, program.stats.observedSteps, program.stats.invalidations], [true, 0, 0]);
  assert.equal(program.rejects(41, ['plan-a']), null);
  assert.equal((await program.run(41, 9))[0], 9 + 41000);
  assert.equal(word(res.kv, 41 * 4), 9 + 41000);
  // An earlier position than the build steps is checked the same way.
  program.beginGeneration();
  assert.equal(await observeStep(program, device, res, 3, 5, {}, 'verify'), 5 + 3000);
  assert.equal((await program.run(4, 6))[0], 6 + 4000);
  program.destroy();
}

// A word that counts decode steps is affine inside a generation; a generation that starts
// at another offset puts it off the line, so the kept program is dropped and the steps
// are observed again. A changed resource is refused the same way.
{
  const device = makeDevice();
  const res = makeModel(device);
  const program = new WholeTokenProgram(device);
  for (const pos of [10, 11, 12, 13]) await observeStep(program, device, res, pos, 42, { stepBase: 9 });
  program.beginGeneration();
  assert.equal(await observeStep(program, device, res, 31, 7, { stepBase: 30 }, 'verify'), 7 + 31000);
  assert.equal(program.ready, false);
  assert.match(program.stats.lastInvalidation, /kept program does not match position 31: .*word 1/);
  assert.equal(program.stats.observedSteps, 1);
  for (const pos of [32, 33, 34]) await observeStep(program, device, res, pos, 7, { stepBase: 30 });
  assert.equal(program.ready, true);
  assert.equal((await program.run(35, 2))[0], 2 + 35000);
  program.beginGeneration();
  await observeStep(program, device, res, 50, 7, { stepBase: 30, identity: ['plan-b'] }, 'verify');
  assert.equal(program.ready, false);
  assert.match(program.stats.lastInvalidation, /execution plan or a bound resource changed/);
  await assert.rejects(observeStep(program, device, res, 51, 7, {}, 'verify'), /No kept program awaits verification/);
  program.destroy();
}

console.log('whole-token-program-contract.test: ok');
