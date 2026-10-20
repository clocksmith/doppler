import { quantizeW8A8Reference, matmulW8A8Reference, rescaleW8A8Reference } from './reference/w8a8.js';

export async function auditW8A8(operators = null) {
  const { initDevice, getKernelCapabilities, destroyDevice } = await import('/src/gpu/device.js');
  const { acquireBuffer, releaseBuffer, readBuffer, getBufferPool, destroyBufferPool } = await import('/src/memory/buffer-pool.js');
  const { unifiedKernelWrapper } = await import('/src/gpu/kernels/kernel-execution.js');
  const { getKernelConfig } = await import('/src/gpu/kernels/kernel-configs.js');
  const { calculateMatmulDispatch } = await import('/src/gpu/kernels/matmul-dispatch.js');
  const { runW8A8Matmul, recordW8A8Matmul, assertW8A8Status, validateW8A8Matmul } = await import('/src/gpu/kernels/w8a8.js');
  const { createTensor } = await import('/src/gpu/tensor.js');
  const { createWeightBuffer } = await import('/src/gpu/weight-buffer.js');
  const { CommandRecorder } = await import('/src/gpu/command-recorder.js');
  const device = await initDevice(), capabilities = getKernelCapabilities();
  if (/software|swiftshader|llvmpipe/i.test(JSON.stringify(capabilities.adapterInfo))) throw Error('Physical GPU required');
  const errors = [], rows = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  const check = (value, message) => { if (!value) throw Error(message); };
  const equal = (a, b, message) => {
    check(a.length === b.length, `${message}: length`);
    for (let i = 0; i < a.length; i++) check(Object.is(a[i], b[i]), `${message}: index${i}, actual${a[i]}, expected${b[i]}`);
  };
  const packedAvailable = capabilities.wgslLanguageFeatures?.includes('packed_4x8_integer_dot_product');
  const variants = ['w8a8_portable', ...(packedAvailable ? ['w8a8_packed', 'w8a8_packed_cooperative'] : [])];
  const settle = async () => {
    for (let i = 0; i < 32; i++) {
      await device.queue.onSubmittedWorkDone();
      const resources = getBufferPool().getStats().resources;
      if (resources.deferredCleanup.bytes === 0) return resources;
    }
    throw Error('Owned GPU retirement did not settle');
  };
  const floats = (size, salt) => Float32Array.from({ length: size }, (_, i) => ((i * 37 + salt) % 255 - 127) / 8);
  const shapes = [[1, 1, 1], [2, 3, 5], [3, 7, 9], [2, 5, 4], [5, 3, 257], [2, 8, 640], [3, 17, 2048], [1, 2, 133144]];
  if (operators) {
    check(Array.isArray(operators.M) && Array.isArray(operators.projections)
      && Number.isSafeInteger(operators.seed), 'Malformed frozen operator suite');
    for (const M of operators.M) for (const { N, K } of operators.projections) {
      check([M, N, K].every(value => Number.isSafeInteger(value) && value > 0), 'Invalid frozen projection geometry');
      shapes.push([M, N, K]);
    }
  }
  try {
    for (const [M, N, K] of shapes) {
      const A = K === 133144 ? new Float32Array(K).fill(127) : floats(M * K, operators?.seed ?? 7);
      const W = K === 133144 ? new Float32Array(N * K).fill(127) : floats(N * K, (operators?.seed ?? 7) + 12);
      if (K === 5) { A.fill(0, 0, K); W.fill(0, 0, K); }
      if (K === 9) A.set([127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0, 126.5]);
      if (K === 4) A.set([0.001003009034320712, 0.00001184656321129296, 0, 0]);
      const qa = quantizeW8A8Reference(A, M, K), qw = quantizeW8A8Reference(W, N, K);
      const expected = matmulW8A8Reference(qa.bytes, qw.bytes, M, N, K);
      const owned = [];
      const alloc = (size, label) => { const value = acquireBuffer(size, undefined, label); owned.push(value); return value; };
      const upload = (data, label) => { const value = alloc(data.byteLength, label); device.queue.writeBuffer(value, 0, data); return value; };
      try {
        const input = createTensor(upload(A, 'audit_input'), 'f32', [M, K]);
        const metadata = { packing: 'w8a8-row-v1', paddedK: qw.paddedK, rowStrideBytes: qw.rowStride, scaleOffsetBytes: 0, dataOffsetBytes: 4 };
        const weights = createWeightBuffer(upload(qw.bytes, 'audit_weights'), 'w8a8', 'row', [N, K], 'audit', null, metadata);
        const packed = alloc(qa.bytes.length, 'audit_packed'), accumulator = alloc(M * N * 4, 'audit_integer');
        const status = upload(new Uint32Array(1), 'audit_status');
        const uniforms = { rows: M, columns: N, inner: K, row_words: qa.rowStride / 4 };
        await unifiedKernelWrapper('w8a8_quantize', null, 'f32', [input.buffer, packed, status], uniforms, M);
        await assertW8A8Status(status);
        equal(new Uint8Array(await readBuffer(packed, qa.bytes.length)), qa.bytes, 'Exact activation bytes');
        for (const variant of variants) {
          const dispatch = calculateMatmulDispatch(variant, false, false, M, N, getKernelConfig('matmul', variant)).workgroups;
          await unifiedKernelWrapper('matmul', null, variant, [packed, weights.buffer, accumulator, status], uniforms, dispatch);
          await assertW8A8Status(status);
          equal(new Int32Array(await readBuffer(accumulator, M * N * 4)), expected.accumulators, 'Exact signed accumulation');
          const options = { variant, outputDtype: 'f32' };
          const result = await runW8A8Matmul(input, weights, M, N, K, options); owned.push(result.buffer);
          equal(new Uint32Array(await readBuffer(result.buffer, M * N * 4)), new Uint32Array(expected.output.buffer), 'Immediate rescaled bits');
          const recorder = new CommandRecorder(device, 'w8a8_audit');
          let recorded;
          try {
            recorded = await recordW8A8Matmul(recorder, input, weights, M, N, K, options); owned.push(recorded.buffer);
            await recorder.submitAndWait();
            equal(new Uint32Array(await readBuffer(recorded.buffer, M * N * 4)), new Uint32Array(expected.output.buffer), 'Recorded rescaled bits');
          } finally { await recorder.abort(); }
          rows.push({ M, N, K, variant, packedBytesExact: true, accumulatorsExact: true, outputBitsExact: true });
        }
      } finally { for (const buffer of owned) releaseBuffer(buffer); }
      check((await settle()).active.bytes === 0, 'Shape retained active GPU buffers');
    }
    // Exercise every byte value in every packed position, including -128 and
    // dirty tails. A malformed word must be skipped and set the sticky status.
    for (const K of [1, 2, 3, 4, 5, 7, 8]) {
      const N = 1024, words = 1 + Math.ceil(K / 4), owned = [];
      const upload = data => {
        const buffer = acquireBuffer(data.byteLength); owned.push(buffer);
        device.queue.writeBuffer(buffer, 0, data); return buffer;
      };
      try {
        const a = new Uint8Array(words * 4), w = new Uint8Array(N * words * 4);
        new DataView(a.buffer).setFloat32(0, 1, true); a.fill(1, 4, 4 + K);
        const expected = new Int32Array(N), view = new DataView(w.buffer);
        for (let column = 0; column < N; column++) {
          const base = column * words * 4;
          view.setFloat32(base, 1, true);
          w.fill(127, base + 4, base + 4 + K);
          w[base + (words - 1) * 4 + (column % 4)] = Math.floor(column / 4);
          for (let word = 0; word < words - 1; word++) {
            let sum = 0, valid = true;
            for (let byte = 0; byte < 4; byte++) {
              const offset = word * 4 + byte, value = w[base + 4 + offset];
              if (value === 128 || (offset >= K && value !== 0)) valid = false;
              if (offset < K) sum += value < 128 ? value : value - 256;
            }
            if (valid) expected[column] += sum;
          }
        }
        const packed = upload(a), weights = upload(w), output = upload(new Int32Array(N)), status = upload(new Uint32Array(1));
        for (const variant of variants) {
          device.queue.writeBuffer(status, 0, new Uint32Array(1));
          const dispatch = calculateMatmulDispatch(variant, false, false, 1, N, getKernelConfig('matmul', variant)).workgroups;
          await unifiedKernelWrapper('matmul', null, variant, [packed, weights, output, status],
            { rows: 1, columns: N, inner: K, row_words: words }, dispatch);
          equal(new Int32Array(await readBuffer(output, N * 4)), expected, 'Packed byte-domain and padding accumulation');
          check(new Uint32Array(await readBuffer(status, 4))[0] === 8, 'Packed byte-domain rejection flags');
          rows.push({ K, variant, byteDomainCases: N, accumulatorsExact: true, statusBits: 8 });
        }
      } finally { for (const buffer of owned) releaseBuffer(buffer); }
    }
    // Scalar conversion ties precede independently checked floating rescale boundaries.
    const integerValues = new Int32Array([16777217, 16777219, -16777217, 2147479576, 0]);
    const scales = [1, 1, 1, 2 ** -126, 1];
    const owned = [];
    const upload = data => { const buffer = acquireBuffer(data.byteLength); owned.push(buffer); device.queue.writeBuffer(buffer, 0, data); return buffer; };
    try {
      const scaleRows = new Uint32Array(scales.length * 2), view = new DataView(scaleRows.buffer);
      scales.forEach((scale, index) => view.setFloat32(index * 8, scale, true));
      const input = upload(integerValues), packed = upload(scaleRows), output = upload(new Uint32Array(scales.length)), status = upload(new Uint32Array(1));
      await unifiedKernelWrapper('w8a8_rescale', null, 'activation', [input, packed, output, status],
        { rows: scales.length, columns: 1, inner: 1, row_words: 2 }, 1);
      await assertW8A8Status(status);
      const expectedFloats = Float32Array.from(integerValues, (value, i) => rescaleW8A8Reference(value, scales[i], 1).output);
      equal(new Uint32Array(await readBuffer(output, scales.length * 4)), new Uint32Array(expectedFloats.buffer), 'Integer conversion ties');
    } finally { for (const buffer of owned) releaseBuffer(buffer); }
    check((await settle()).active.bytes === 0, 'Rescale retained active GPU buffers');
    const fromBits = bits => new Float32Array(new Uint32Array([bits]).buffer)[0];
    const pairs = [[2 ** -126, 0.5], [-(2 ** -126), 0.5], [2 ** -126, fromBits(0x3f7fffff)],
      [fromBits(0x7f7fffff), 1], [fromBits(0x7f7fffff), fromBits(0x3f800001)],
      [fromBits(0x3fffffff), fromBits(0x3fffffff)], [-0, 1], [0, 1]];
    let random = 0x52f781a3;
    const next = () => { random = (Math.imul(random, 1664525) + 1013904223) >>> 0; return random; };
    for (let i = 0; i < 1024; i++) {
      const a = (next() & 0x807fffff) | ((1 + next() % 254) << 23);
      const b = (next() & 0x007fffff) | ((1 + next() % 254) << 23);
      pairs.push([fromBits(a), fromBits(b)]);
    }
    const productInput = Float32Array.from(pairs, pair => pair[0]), productOutput = new Float32Array(pairs.length);
    const scaleRows = new Uint32Array(pairs.length * 2), scaleView = new DataView(scaleRows.buffer);
    let overflows = 0;
    pairs.forEach(([value, scale], i) => {
      scaleView.setFloat32(i * 8, scale, true);
      if (value === 0) { productOutput[i] = value; return; }
      try { productOutput[i] = rescaleW8A8Reference(value < 0 ? -1 : 1, Math.abs(value), scale).output; }
      catch (error) { check(/overflow/i.test(error.message), error.message); overflows++; productOutput[i] = 0; }
    });
    const productsOwned = [];
    const productUpload = data => { const buffer = acquireBuffer(data.byteLength); productsOwned.push(buffer); device.queue.writeBuffer(buffer, 0, data); return buffer; };
    try {
      const input = productUpload(productInput), scaleBuffer = productUpload(scaleRows);
      const output = productUpload(new Uint32Array(pairs.length)), status = productUpload(new Uint32Array(1));
      await unifiedKernelWrapper('w8a8_rescale', null, 'weight', [input, scaleBuffer, output, status],
        { rows: 1, columns: pairs.length, inner: 1, row_words: 2 }, Math.ceil(pairs.length / 256));
      const statusBits = new Uint32Array(await readBuffer(status, 4))[0];
      check(statusBits === (overflows ? 16 : 0), 'Rescale overflow status differs');
      equal(new Uint32Array(await readBuffer(output, pairs.length * 4)), new Uint32Array(productOutput.buffer), 'Exact normal-pair multiplication including signed FTZ');
      rows.push({ rescalePairs: pairs.length, overflows, exactBits: true, minimumNormalRoundUp: true, signedZeroFlush: true });
    } finally { for (const buffer of productsOwned) releaseBuffer(buffer); }
    check((await settle()).active.bytes === 0, 'Product controls retained active GPU buffers');
    for (const test of [
      { name: 'nan', A: new Float32Array([NaN]), flag: 1 },
      { name: 'infinity', A: new Float32Array([Infinity]), flag: 1 },
      { name: 'subnormal', A: new Float32Array([2 ** -149]), flag: 2 },
      { name: 'scale-underflow', A: new Float32Array([2 ** -126]), flag: 4 },
      { name: 'negative128', A: new Float32Array([1]), flag: 8, mutate: bytes => { bytes[4] = 128; } },
      { name: 'padding', A: new Float32Array([1]), flag: 8, mutate: bytes => { bytes[5] = 1; } },
      { name: 'bad-scale', A: new Float32Array([1]), flag: 4, mutate: bytes => { new DataView(bytes.buffer).setFloat32(0, Infinity, true); } },
      { name: 'zero-row-scale', A: new Float32Array([1]), W: new Float32Array([0]), flag: 4,
        mutate: bytes => { new DataView(bytes.buffer).setFloat32(0, 2, true); } },
      { name: 'overflow', A: new Float32Array([2 ** 120]), W: new Float32Array([2 ** 120]), flag: 16 },
    ]) {
      const owned = [];
      const upload = data => { const value = acquireBuffer(data.byteLength); owned.push(value); device.queue.writeBuffer(value, 0, data); return value; };
      try {
        const q = quantizeW8A8Reference(test.W ?? new Float32Array([1]), 1, 1); test.mutate?.(q.bytes);
        const input = createTensor(upload(test.A), 'f32', [1, 1]);
        const weights = createWeightBuffer(upload(q.bytes), 'w8a8', 'row', [1, 1], 'invalid', null,
          { packing: 'w8a8-row-v1', paddedK: 4, rowStrideBytes: 8, scaleOffsetBytes: 0, dataOffsetBytes: 4 });
        for (const variant of variants) for (const recorded of [false, true]) {
          const recorder = recorded ? new CommandRecorder(device, 'w8a8_failure') : null;
          let rejected = null;
          try {
            const result = recorded ? await recordW8A8Matmul(recorder, input, weights, 1, 1, 1, { variant, outputDtype: 'f32' })
              : await runW8A8Matmul(input, weights, 1, 1, 1, { variant, outputDtype: 'f32' });
            owned.push(result.buffer);
            if (recorder) await recorder.submitAndWait();
          } catch (error) { rejected = error; }
          finally { await recorder?.abort(); }
          check(rejected?.code === 'DOPPLER_W8A8_INVALID_INPUT' && (rejected.statusBits & test.flag), `Missing ${test.name} rejection`);
          rows.push({ failure: test.name, variant, recorded, code: rejected.code, statusBits: rejected.statusBits });
        }
      } finally { for (const buffer of owned) releaseBuffer(buffer); }
      check((await settle()).active.bytes === 0, 'Rejected input retained active GPU buffers');
    }
    // One sticky status across several projections survives partial submissions and
    // deferred completion. Abandoning an unsubmitted recorder also settles its scope.
    const lifecycleOwned = [];
    const lifecycleUpload = data => { const b = acquireBuffer(data.byteLength); lifecycleOwned.push(b); device.queue.writeBuffer(b, 0, data); return b; };
    try {
      const q = quantizeW8A8Reference(new Float32Array([1]), 1, 1);
      const input = createTensor(lifecycleUpload(new Float32Array([1])), 'f32', [1, 1]);
      const invalid = createTensor(lifecycleUpload(new Float32Array([NaN])), 'f32', [1, 1]);
      const weights = createWeightBuffer(lifecycleUpload(q.bytes), 'w8a8', 'row', [1, 1], 'lifecycle', null,
        { packing: 'w8a8-row-v1', paddedK: 4, rowStrideBytes: 8, scaleOffsetBytes: 0, dataOffsetBytes: 4 });
      const options = { variant: 'w8a8_portable', outputDtype: 'f32', constants: { WORKGROUP_SIZE: 64 } };
      const invalidArguments = [
        () => validateW8A8Matmul(device, input, weights, 0, 1, 1, options),
        () => validateW8A8Matmul(device, input, weights, 1.5, 1, 1, options),
        () => validateW8A8Matmul(device, input, weights, 1, 1, 133145, options),
        () => validateW8A8Matmul(device, input, weights, 1, 2, 1, options),
        () => validateW8A8Matmul(device, input, { ...weights, metadata: { ...weights.metadata, paddedK: 8 } }, 1, 1, 1, options),
        () => validateW8A8Matmul(device, input, weights, 1, 1, 1, { ...options, variant: undefined }),
        () => validateW8A8Matmul(device, input, weights, 1, 1, 1, { ...options, aOffset: 1 }),
        () => validateW8A8Matmul(device, input, weights, 1, 1, 1, { ...options, outputBuffer: input.buffer }),
        () => validateW8A8Matmul(device, input, weights, 1, 1, 1, { ...options, constants: { WORKGROUP_SIZE: 32 } }),
      ];
      for (const invalidCall of invalidArguments) {
        let rejected = false; try { invalidCall(); } catch { rejected = true; }
        check(rejected, 'Invalid preflight arguments admitted');
      }
      for (const partial of [false, true]) {
        const recorder = new CommandRecorder(device, 'shared_w8a8_status');
        let validations = 0;
        const enqueue = recorder.enqueueValidationTask.bind(recorder);
        recorder.enqueueValidationTask = task => { validations++; enqueue(task); };
        try {
          for (const operand of [invalid, input]) {
            const result = await recordW8A8Matmul(recorder, operand, weights, 1, 1, 1, options);
            lifecycleOwned.push(result.buffer);
            if (partial) recorder.submitPartial();
          }
          check(validations === 1, 'Multiple status readbacks for one recorder');
          recorder.submit({ cleanup: 'deferred' });
          let rejected;
          try { await recorder.completeDeferredCleanup(); } catch (error) { rejected = error; }
          check(rejected?.code === 'DOPPLER_W8A8_INVALID_INPUT' && rejected.statusBits === 1,
            'Deferred completion lost earlier projection failure');
          rows.push({ lifecycle: 'shared-status-deferred', partial, validations, code: rejected.code });
        } finally { await recorder.abort(); }
      }
      const abandoned = new CommandRecorder(device, 'abandoned_w8a8');
      const output = await recordW8A8Matmul(abandoned, input, weights, 1, 1, 1, options);
      lifecycleOwned.push(output.buffer);
      await abandoned.abort();
      const controller = new AbortController(); controller.abort();
      let aborted;
      try { await runW8A8Matmul(input, weights, 1, 1, 1, { ...options, signal: controller.signal }); }
      catch (error) { aborted = error; }
      check(aborted?.name === 'AbortError', 'Pre-dispatch cancellation ignored');
      const result = await runW8A8Matmul(input, weights, 1, 1, 1, options);
      lifecycleOwned.push(result.buffer);
      const reference = matmulW8A8Reference(q.bytes, q.bytes, 1, 1, 1);
      equal(new Uint32Array(await readBuffer(result.buffer, 4)), new Uint32Array(reference.output.buffer), 'Fresh reuse after failure and abort');
      rows.push({ lifecycle: 'abort-and-fresh-reuse', preflightRejections: invalidArguments.length, outputExact: true });
    } finally { for (const buffer of lifecycleOwned) releaseBuffer(buffer); }
    check((await settle()).active.bytes === 0, 'Lifecycle controls retained active GPU buffers');
    check(errors.length === 0, 'WebGPU uncaptured errors: ' + errors.join('; '));
    return { qualified: true, capabilities, packedAvailable, rows, resources: await settle() };
  } catch (error) { return { qualified: false, capabilities, packedAvailable, rows, failure: error.stack, errors }; }
  finally { await settle(); destroyBufferPool(); destroyDevice(); }
}
