import { quantizeW8A8Reference } from './reference/w8a8.js';

// Operator tuning evidence, not model qualification. Timestamp instrumentation
// runs separately from the primary, alternating wall-clock samples.
export async function profileW8A8(config) {
  const { initDevice, getKernelCapabilities, destroyDevice } = await import('/src/gpu/device.js');
  const { acquireBuffer, releaseBuffer, readBuffer, getBufferPool, destroyBufferPool } = await import('/src/memory/buffer-pool.js');
  const { recordW8A8Matmul } = await import('/src/gpu/kernels/w8a8.js');
  const { createTensor } = await import('/src/gpu/tensor.js');
  const { createWeightBuffer } = await import('/src/gpu/weight-buffer.js');
  const { CommandRecorder } = await import('/src/gpu/command-recorder.js');
  const { computeSampleStats } = await import('/src/debug/stats.js');
  if (!Number.isSafeInteger(config.seed) || !Number.isSafeInteger(config.warmups) || config.warmups < 1
    || !Number.isSafeInteger(config.pairs) || config.pairs < 2
    || !Number.isSafeInteger(config.operationsPerSubmission) || config.operationsPerSubmission < 1
    || !Number.isSafeInteger(config.profileRuns) || config.profileRuns < 1 || !config.variants?.length || !config.shapes?.length) {
    throw Error('Explicit seed, warmups, paired runs, variants and shapes required');
  }
  const device = await initDevice();
  const report = { capabilities: getKernelCapabilities(), config, shapes: [], errors: [] };
  device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
  let random = config.seed >>> 0;
  const floats = length => Float32Array.from({ length }, () => {
    random = (Math.imul(random, 1664525) + 1013904223) >>> 0;
    return ((random >>> 8) % 255 - 127) / 8;
  });
  try {
    if (/software|swiftshader|llvmpipe/i.test(JSON.stringify(report.capabilities.adapterInfo))) throw Error('Physical GPU required');
    for (const [M, N, K] of config.shapes) {
      if (![M, N, K].every(value => Number.isSafeInteger(value) && value > 0)) throw Error('Invalid shape');
      const owned = [];
      const upload = data => {
        const buffer = acquireBuffer(data.byteLength); owned.push(buffer);
        device.queue.writeBuffer(buffer, 0, data); return buffer;
      };
      try {
        const input = createTensor(upload(floats(M * K)), 'f32', [M, K]);
        const q = quantizeW8A8Reference(floats(N * K), N, K);
        const weights = createWeightBuffer(upload(q.bytes), 'w8a8', 'row', [N, K], 'profile', null,
          { packing: 'w8a8-row-v1', paddedK: q.paddedK, rowStrideBytes: q.rowStride, scaleOffsetBytes: 0, dataOffsetBytes: 4 });
        const output = upload(new Uint32Array(M * N));
        const row = { M, N, K, runs: [], profiles: [], summaries: {} }; report.shapes.push(row);
        let baseline;
        const execute = async (variant, profile) => {
          const recorder = new CommandRecorder(device, 'w8a8_profile', { profile });
          try {
            const start = performance.now();
            for (let operation = 0; operation < config.operationsPerSubmission; operation++) {
              await recordW8A8Matmul(recorder, input, weights, M, N, K,
                { variant, outputDtype: 'f32', outputBuffer: output });
            }
            const recorded = performance.now();
            await recorder.submitAndWait();
            const finished = performance.now();
            const gpuMs = profile ? await recorder.resolveProfileTimings({ requireComplete: true }) : null;
            const bits = new Uint32Array(await readBuffer(output, M * N * 4));
            if (!baseline) baseline = bits;
            else if (bits.some((value, index) => value !== baseline[index])) throw Error(`${variant}: output bits differ`);
            return { variant, wallMs: finished - start, recordMs: recorded - start,
              submitAndValidateMs: finished - recorded, gpuMs, exactOutput: true };
          } finally { await recorder.abort(); }
        };
        for (let i = -config.warmups; i < config.pairs; i++) {
          const order = Math.abs(i) % 2 ? [...config.variants].reverse() : config.variants;
          for (const variant of order) row.runs.push({ pair: i, warmup: i < 0, ...await execute(variant, false) });
        }
        for (const variant of config.variants) {
          for (let i = 0; i < config.profileRuns; i++) row.profiles.push(await execute(variant, true));
          row.summaries[variant] = computeSampleStats(row.runs.filter(run => !run.warmup && run.variant === variant)
            .map(run => run.wallMs), { outlierPolicy: 'none' });
        }
      } finally {
        await device.queue.onSubmittedWorkDone();
        for (const buffer of owned) releaseBuffer(buffer);
      }
    }
    if (report.errors.length) throw Error(report.errors.join('\n'));
    await device.queue.onSubmittedWorkDone();
    report.resources = getBufferPool().getStats().resources;
    if (report.resources.active.bytes || report.resources.deferredCleanup.bytes) throw Error('Profiling retained live GPU buffers');
    report.passed = true;
    return report;
  } catch (error) {
    report.passed = false;
    report.failure = error.stack;
    return report;
  } finally { destroyBufferPool(); destroyDevice(); }
}
