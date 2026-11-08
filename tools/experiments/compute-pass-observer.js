// Diagnostic adapter only. Does not split passes or replace pipelines/dispatches.
// Records encoded passes separately from submitted GPU work. Existing timestamp owners fail closed.
export function observeComputePasses(device, options) {
  const { queriesPerBlock, maxPasses } = options;
  if (!device.features.has('timestamp-query')) throw new Error('Pass observer requires timestamp-query');
  if (!Number.isSafeInteger(queriesPerBlock) || queriesPerBlock < 2 || queriesPerBlock % 2
      || !Number.isSafeInteger(maxPasses) || maxPasses < 1) throw new Error('Invalid pass timestamp budgets');
  // Node providers can expose native instances without publishing constructor globals.
  // An empty, unsubmitted probe discovers the owning provider's actual prototypes.
  const probeEncoder = device.createCommandEncoder({ label: 'diagnostic-prototype-probe' });
  const probePass = probeEncoder.beginComputePass({});
  const encoderProto = Object.getPrototypeOf(probeEncoder), passProto = Object.getPrototypeOf(probePass);
  probePass.end();
  probeEncoder.finish();
  const begin = encoderProto.beginComputePass, pipeline = passProto.setPipeline;
  const dispatch = passProto.dispatchWorkgroups, indirect = passProto.dispatchWorkgroupsIndirect;
  const finish = encoderProto.finish, queue = device.queue, submit = queue.submit;
  const submitDescriptor = Object.getOwnPropertyDescriptor(queue, 'submit');
  const blocks = [], records = [], passes = new WeakMap();
  const encoderRecords = new WeakMap(), commandBufferRecords = new WeakMap();
  let active = true, disposed = false, collected = false;
  function stopCapture() {
    active = false;
    encoderProto.beginComputePass = begin;
    passProto.setPipeline = pipeline;
    passProto.dispatchWorkgroups = dispatch;
    passProto.dispatchWorkgroupsIndirect = indirect;
  }
  function restore() {
    stopCapture();
    encoderProto.finish = finish;
    if (submitDescriptor) Object.defineProperty(queue, 'submit', submitDescriptor);
    else delete queue.submit;
  }
  encoderProto.finish = function (...args) {
    const buffer = finish.apply(this, args);
    commandBufferRecords.set(buffer, encoderRecords.get(this) ?? []);
    return buffer;
  };
  queue.submit = function (buffers) {
    const batch = Array.from(buffers);
    const result = submit.call(this, batch);
    for (const buffer of batch) {
      for (const record of commandBufferRecords.get(buffer) ?? []) record.submitted = true;
    }
    return result;
  };
  function destroy() {
    for (const block of blocks) {
      if (block.readback?.mapState === 'mapped') block.readback.unmap();
      block.readback?.destroy(); block.resolve?.destroy(); block.queries.destroy();
    }
  }
  encoderProto.beginComputePass = function (descriptor = {}) {
    if (!active) return begin.call(this, descriptor);
    if (descriptor.timestampWrites) throw new Error('Pass already has a timestamp owner');
    if (records.length >= maxPasses) throw new Error('Pass timestamp budget exhausted');
    let block = blocks.at(-1);
    if (!block || block.n + 2 > queriesPerBlock) {
      block = { queries: device.createQuerySet({ type: 'timestamp', count: queriesPerBlock }), n: 0, index: blocks.length };
      blocks.push(block);
    }
    const q = block.n; block.n += 2;
    const pass = begin.call(this, { ...descriptor, timestampWrites: {
      querySet: block.queries, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1,
    } });
    const record = { submitted: false, block: block.index, q, label: descriptor.label ?? '', pipelines: [], direct: [], indirect: [] };
    records.push(record); passes.set(pass, record);
    const group = encoderRecords.get(this) ?? [];
    group.push(record); encoderRecords.set(this, group);
    return pass;
  };
  passProto.setPipeline = function (value) {
    const record = passes.get(this);
    if (record) record.pipelines.push(value.label || '(unlabelled pipeline)');
    return pipeline.call(this, value);
  };
  passProto.dispatchWorkgroups = function (...size) {
    passes.get(this)?.direct.push(size);
    return dispatch.apply(this, size);
  };
  passProto.dispatchWorkgroupsIndirect = function (buffer, offset) {
    passes.get(this)?.indirect.push({ label: buffer.label ?? '', offset });
    return indirect.call(this, buffer, offset);
  };
  return {
    stop: stopCapture,
    async collect() {
      if (collected || disposed) throw new Error('Pass observer already consumed');
      collected = true; stopCapture();
      try {
        const encoder = device.createCommandEncoder({ label: 'diagnostic-pass-timestamp-resolution' });
        for (const block of blocks) {
          const size = block.n * 8;
          block.resolve = device.createBuffer({ size, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
          block.readback = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          encoder.resolveQuerySet(block.queries, 0, block.n, block.resolve, 0);
          encoder.copyBufferToBuffer(block.resolve, 0, block.readback, 0, size);
        }
        device.queue.submit([encoder.finish()]);
        const timestamps = [];
        for (const block of blocks) {
          await block.readback.mapAsync(GPUMapMode.READ);
          timestamps.push(new BigUint64Array(block.readback.getMappedRange().slice(0)));
        }
        let first = null, last = null, sumMs = 0;
        const byPipeline = {};
        for (const record of records) {
          if (!record.submitted) { record.status = 'encoded-not-submitted'; continue; }
          record.status = 'submitted';
          const ts = timestamps[record.block], t0 = ts[record.q], t1 = ts[record.q + 1];
          if (t0 === 0n || t1 < t0) throw new Error('Missing or reversed compute-pass timestamps');
          first = first === null || t0 < first ? t0 : first;
          last = last === null || t1 > last ? t1 : last;
          record.ms = Number(t1 - t0) / 1e6; sumMs += record.ms;
          record.startNs = t0.toString(); record.endNs = t1.toString();
          const name = record.pipelines.join(' -> ') || record.label || '(unlabelled pass)';
          const group = byPipeline[name] ??= { passes: 0, dispatches: 0, ms: 0 };
          group.passes++; group.dispatches += record.direct.length + record.indirect.length; group.ms += record.ms;
        }
        return { diagnosticOnly: true, scope: 'Submitted compute passes begun through stop, or collection when stop is omitted. Encoded but unsubmitted passes are retained without GPU time. No pass splitting.',
          options, capturedPasses: records.length, submittedPasses: records.filter(record => record.submitted).length, unsubmittedPasses: records.filter(record => !record.submitted).length, queryBlocks: blocks.length, sumPassMs: sumMs,
          firstToLastPassSpanMs: first === null ? null : Number(last - first) / 1e6,
          byPipeline, records };
      } finally { restore(); destroy(); disposed = true; }
    },
    dispose() { if (disposed) return; restore(); destroy(); disposed = true; },
  };
}
