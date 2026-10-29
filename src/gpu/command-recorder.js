import { getDevice, hasFeature, FEATURES } from './device.js';
import { allowReadback, trackAllocation } from '../debug/perf-guards.js';
import { getUniformCache, toUniformArrayBuffer } from './uniform-cache.js';
import { isBufferActive, isPersistentBuffer, releaseBuffer, discardBuffer } from '../memory/buffer-pool.js';
import { log } from '../debug/index.js';
import { createRecorderTimestampResources } from './recorder-timestamps.js';
import { getPipelineProvenance } from './kernels/pipeline-cache.js';


export class CommandRecorder {
  device;
  label;
  decodeWorkspace = null;
  #encoder;
  #tempBuffers;
  // Pooled buffers to release (not destroy) after submit
  #pooledBuffers;
  #tempBufferSet;
  #pooledBufferSet;
  #cleanupPromise = null;
  #deferredCleanup = null;
  #completionTasks;
  #completionFailure = null;
  #submitted;
  #submissionCount = 0;
  #opCount;
  #opLabelCounts;
  #dispatchRecords;
  #recordDispatches;
  #aggregateDispatches;
  #dispatchRecordIndex;
  #recordLabels;
  #computePassCount;
  #activeComputePass = null;
  // Profiling state
  #profilingEnabled;
  #profileRequested;
  #profilePhase;
  #profileReport = null;
  #querySet = null;
  #queryBuffer = null;
  #readbackBuffer = null;
  #profileEntries = [];
  #nextQueryIndex = 0;
  #queryCapacity = 0;
  #submitLatencyMs = null;
  #uniformCache = null;

  constructor(device = null, label = 'command_recorder', options = {}) {
    this.device = device || getDevice();
    if (!this.device) {
      throw new Error('[CommandRecorder] No GPU device available');
    }

    this.#profilePhase = null;
    this.setProfilePhase(options.profilePhase ?? null);
    this.label = label;
    this.#encoder = this.device.createCommandEncoder({ label });
    // Temporary buffers to destroy after submit (created directly by recorder)
    this.#tempBuffers = [];
    // Pooled buffers to release after submit (came from buffer pool)
    this.#pooledBuffers = [];
    this.#tempBufferSet = new Set();
    this.#pooledBufferSet = new Set();
    this.#cleanupPromise = null;
    this.#deferredCleanup = null;
    this.#completionTasks = [];

    this.#submitted = false;
    this.#opCount = 0;
    this.#recordLabels = options.recordLabels !== false;
    this.#recordDispatches = options.recordDispatches !== false;
    this.#aggregateDispatches = options.aggregateDispatches === true;
    this.#opLabelCounts = this.#recordLabels ? Object.create(null) : null;
    this.#dispatchRecords = this.#recordDispatches ? [] : null;
    this.#dispatchRecordIndex = this.#recordDispatches && this.#aggregateDispatches
      ? new Map()
      : null;
    this.#computePassCount = 0;
    this.#activeComputePass = null;
    this.#profileRequested = options.profile === true;
    this.#profilingEnabled = this.#profileRequested && hasFeature(FEATURES.TIMESTAMP_QUERY);
    if (this.#profilingEnabled) {
      this.#initProfiling();
    }
  }

  #initProfiling() {
    try {
      const resources = createRecorderTimestampResources(this.device, this.label);
      this.#querySet = resources.querySet;
      this.#queryBuffer = resources.queryBuffer;
      this.#readbackBuffer = resources.readbackBuffer;
      this.#queryCapacity = resources.capacity;
    } catch (error) {
      log.warn('CommandRecorder', `Failed to initialize profiling: ${error}`);
      this.#profilingEnabled = false;
    }
  }

  // Phase is caller-owned observation metadata, never inferred from a kernel label.
  setProfilePhase(phase) {
    if (phase !== null && phase !== 'prefill' && phase !== 'decode') {
      throw new Error('[CommandRecorder] profilePhase must be prefill, decode, or null.');
    }
    if (this.#submitted) throw new Error('[CommandRecorder] Cannot change phase after submit');
    this.#profilePhase = phase;
  }

  isProfilingEnabled() {
    return this.#profilingEnabled;
  }

  createTempBuffer(size, usage, label = 'temp_buffer') {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot create buffers after submit');
    }

    const buffer = this.device.createBuffer({
      label: `${this.label}_${label}_${this.#tempBuffers.length}`,
      size,
      usage,
    });
    trackAllocation(size, label);

    this.#tempBuffers.push(buffer);
    this.#tempBufferSet.add(buffer);
    return buffer;
  }

  createIndirectDispatchBuffer(
    workgroups = [0, 0, 0],
    label = 'indirect_dispatch'
  ) {
    const data = workgroups instanceof Uint32Array
      ? workgroups
      : new Uint32Array(workgroups);
    const size = Math.max(12, data.byteLength);
    const buffer = this.createTempBuffer(
      size,
      GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label
    );
    const source =  (data.buffer);
    this.device.queue.writeBuffer(buffer, 0, source, data.byteOffset, data.byteLength);
    return buffer;
  }

  writeIndirectDispatchBuffer(
    buffer,
    workgroups,
    offset = 0
  ) {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot write buffers after submit');
    }
    const data = workgroups instanceof Uint32Array
      ? workgroups
      : new Uint32Array(workgroups);
    const source =  (data.buffer);
    this.device.queue.writeBuffer(buffer, offset, source, data.byteOffset, data.byteLength);
  }

  createUniformBuffer(data, label = 'uniforms') {
    this.#uniformCache ??= getUniformCache(this.device);
    return this.#uniformCache.getOrCreate(toUniformArrayBuffer(data), label);
  }

  #normalizeOperationLabel(label) {
    return typeof label === 'string' && label.length > 0
      ? label
      : 'compute_pass';
  }

  #recordOperation(label) {
    const opLabel = this.#normalizeOperationLabel(label);
    this.#opCount++;
    if (this.#recordLabels) {
      this.#opLabelCounts[opLabel] = (this.#opLabelCounts[opLabel] ?? 0) + 1;
    }
    return opLabel;
  }

  #recordDispatchOperation(label) {
    this.#opCount++;
    if (!this.#recordLabels && !this.#profilingEnabled) {
      return null;
    }

    const opLabel = this.#normalizeOperationLabel(label);
    if (this.#recordLabels) {
      this.#opLabelCounts[opLabel] = (this.#opLabelCounts[opLabel] ?? 0) + 1;
    }
    return opLabel;
  }

  #recordDispatchGeometry(label, kind, workgroups) {
    if (!this.#recordDispatches) {
      return;
    }
    const normalizedWorkgroups = workgroups === null
      ? null
      : [workgroups[0], workgroups[1], workgroups[2]];
    if (!this.#aggregateDispatches) {
      this.#dispatchRecords.push({
        label,
        kind,
        workgroups: normalizedWorkgroups,
      });
      return;
    }
    const geometryKey = normalizedWorkgroups === null
      ? 'indirect'
      : normalizedWorkgroups.join(',');
    const key = `${label}\u0000${kind}\u0000${geometryKey}`;
    const existingIndex = this.#dispatchRecordIndex.get(key);
    if (existingIndex !== undefined) {
      this.#dispatchRecords[existingIndex].count += 1;
      return;
    }
    this.#dispatchRecordIndex.set(key, this.#dispatchRecords.length);
    this.#dispatchRecords.push({
      label,
      kind,
      workgroups: normalizedWorkgroups,
      count: 1,
    });
  }

  #beginRawComputePass(opLabel, dispatch = null) {
    this.#computePassCount++;
    const passLabel = `${this.label}_${opLabel}_${this.#computePassCount}`;

    if (this.#profilingEnabled && this.#querySet && this.#nextQueryIndex + 2 <= this.#queryCapacity) {
      const startIndex = this.#nextQueryIndex;
      const endIndex = startIndex + 1;
      this.#nextQueryIndex += 2;

      this.#profileEntries.push({
        label: opLabel,
        phase: this.#profilePhase,
        dispatch,
        startQueryIndex: startIndex,
        endQueryIndex: endIndex,
      });

      return this.#encoder.beginComputePass({
        label: passLabel,
        timestampWrites: {
          querySet: this.#querySet,
          beginningOfPassWriteIndex: startIndex,
          endOfPassWriteIndex: endIndex,
        },
      });
    }

    return this.#encoder.beginComputePass({
      label: passLabel,
    });
  }

  closeActiveComputePass() {
    if (!this.#activeComputePass) {
      return;
    }
    const pass = this.#activeComputePass;
    this.#activeComputePass = null;
    pass.end();
  }

  #ensureActiveComputePass() {
    if (!this.#activeComputePass) {
      this.#activeComputePass = this.#beginRawComputePass('coalesced_compute');
    }
    return this.#activeComputePass;
  }

  beginComputePass(label = 'compute_pass') {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot begin pass after submit');
    }
    this.closeActiveComputePass();
    return this.#beginRawComputePass(this.#recordOperation(label));
  }

  recordDispatch(pipeline, bindGroup, workgroups, label = 'compute') {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot record dispatch after submit');
    }
    const opLabel = this.#recordDispatchOperation(label);
    this.#recordDispatchGeometry(
      opLabel ?? this.#normalizeOperationLabel(label),
      'direct',
      workgroups
    );
    if (this.#profilingEnabled) {
      const pass = this.#beginRawComputePass(opLabel, {
        kind: 'direct', workgroups: [...workgroups], pipelineLabel: pipeline.label || null,
        pipeline: getPipelineProvenance(pipeline),
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
      pass.end();
      return;
    }
    const pass = this.#ensureActiveComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }

  recordDispatchIndirect(pipeline, bindGroup, indirectBuffer, indirectOffset = 0, label = 'compute') {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot record dispatch after submit');
    }
    const opLabel = this.#recordDispatchOperation(label);
    this.#recordDispatchGeometry(
      opLabel ?? this.#normalizeOperationLabel(label),
      'indirect',
      null
    );
    if (this.#profilingEnabled) {
      const pass = this.#beginRawComputePass(opLabel, {
        kind: 'indirect', workgroups: null, pipelineLabel: pipeline.label || null,
        pipeline: getPipelineProvenance(pipeline),
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
      pass.end();
      return;
    }
    const pass = this.#ensureActiveComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
  }

  getEncoder() {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot access encoder after submit');
    }
    this.closeActiveComputePass();
    return this.#encoder;
  }

  trackTemporaryBuffer(buffer) {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot track buffers after submit');
    }
    if (this.#tempBufferSet.has(buffer) || this.#pooledBufferSet.has(buffer)) {
      return;
    }
    if (isPersistentBuffer(buffer)) {
      return;
    }
    if (isBufferActive(buffer)) {
      this.#pooledBufferSet.add(buffer);
      this.#pooledBuffers.push(buffer);
      return;
    }
    // Recorder cleanup owns only recorder-created buffers and pooled buffers that are
    // still active. External raw GPUBuffer instances may be persistent model weights.
    // Treating them as temporary would destroy weights at submit time.
  }

  enqueueCompletionTask(task) {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot enqueue completion tasks after submit');
    }
    if (typeof task !== 'function') {
      throw new Error('[CommandRecorder] completion task must be a function');
    }
    this.#completionTasks.push({ task, required: false });
  }

  enqueueValidationTask(task) {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot enqueue validation tasks after submit');
    }
    if (typeof task !== 'function') {
      throw new Error('[CommandRecorder] validation task must be a function');
    }
    this.#completionTasks.push({ task, required: true });
  }

  async #runCompletionTasks() {
    if (this.#completionTasks.length === 0) {
      return;
    }
    const tasks = this.#completionTasks;
    this.#completionTasks = [];
    for (const { task, required } of tasks) {
      try {
        await task();
      } catch (error) {
        if (required) {
          this.#completionFailure ??= { error };
        } else {
          log.warn('CommandRecorder', `Completion task failed: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }
  }

  #finalizeTrackedBuffers(buffersToDestroy, buffersToRelease, discardPooled) {
    for (const buffer of buffersToDestroy) {
      buffer.destroy();
    }
    for (const buffer of buffersToRelease) {
      if (discardPooled) {
        discardBuffer(buffer);
      } else {
        releaseBuffer(buffer);
      }
    }
    if (this.#submissionCount > 0) this.#uniformCache?.flushPendingDestruction();
  }

  #takeTrackedBuffers() {
    const buffersToDestroy = this.#tempBuffers;
    const buffersToRelease = this.#pooledBuffers;
    this.#tempBuffers = [];
    this.#pooledBuffers = [];
    this.#tempBufferSet.clear();
    this.#pooledBufferSet.clear();
    return { buffersToDestroy, buffersToRelease };
  }

  #finalizeDeferredCleanup(discardPooled = false) {
    if (!this.#deferredCleanup) {
      return;
    }
    const { buffersToDestroy, buffersToRelease, submitStart } = this.#deferredCleanup;
    this.#deferredCleanup = null;
    this.#submitLatencyMs = performance.now() - submitStart;
    this.#finalizeTrackedBuffers(buffersToDestroy, buffersToRelease, discardPooled);
  }

  submitPartial() {
    if (this.#submitted) throw new Error('[CommandRecorder] Already submitted');
    this.closeActiveComputePass();
    try {
      this.device.queue.submit([this.#encoder.finish()]);
      this.#submissionCount += 1;
      this.#encoder = this.device.createCommandEncoder({ label: this.label });
    } catch (error) {
      this.abort();
      throw error;
    }
  }

  #cleanupAbortedBuffers(buffersToDestroy, buffersToRelease) {
    this.#completionTasks = [];
    const cleanup = discard => {
      this.#finalizeTrackedBuffers(buffersToDestroy, buffersToRelease, discard);
      this.#destroyProfilingResources();
    };
    if (this.#submissionCount === 0) {
      cleanup(false);
      return;
    }
    this.#cleanupPromise = this.device.queue.onSubmittedWorkDone()
      .then(() => cleanup(false), () => cleanup(true));
  }

  submit(options = {}) {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Already submitted');
    }
    const cleanup = options.cleanup ?? 'queue';
    if (cleanup !== 'queue' && cleanup !== 'deferred') {
      throw new Error('[CommandRecorder] submit cleanup must be "queue" or "deferred".');
    }

    const submitStart = performance.now();
    this.closeActiveComputePass();
    const { buffersToDestroy, buffersToRelease } = this.#takeTrackedBuffers();
    try {
      this.device.queue.submit([this.#encoder.finish()]);
      this.#submissionCount += 1;
    } catch (error) {
      this.#submitted = true;
      this.#cleanupAbortedBuffers(buffersToDestroy, buffersToRelease);
      throw error;
    }

    this.#submitted = true;
    this.#cleanupPromise = null;

    if (cleanup === 'deferred') {
      if (this.#completionTasks.length > 0) {
        this.#cleanupPromise = this.device.queue.onSubmittedWorkDone().then(async () => {
          await this.#runCompletionTasks();
        }).catch((err) => {
          this.#completionTasks = [];
          this.#completionFailure ??= { error: err };
        });
      }
      this.#deferredCleanup = {
        buffersToDestroy,
        buffersToRelease,
        submitStart,
      };
      return;
    }

    this.#cleanupPromise = this.device.queue.onSubmittedWorkDone().then(async () => {
      await this.#runCompletionTasks();
      this.#submitLatencyMs = performance.now() - submitStart;
      this.#finalizeTrackedBuffers(buffersToDestroy, buffersToRelease, false);
    }).catch((err) => {
      this.#completionTasks = [];
      this.#completionFailure ??= { error: err };
      this.#finalizeTrackedBuffers(buffersToDestroy, buffersToRelease, true);
    });
  }

  async completeDeferredCleanup(options = {}) {
    const discardPooled = options.discardPooled === true;
    if (this.#cleanupPromise) {
      await this.#cleanupPromise;
    }
    this.#finalizeDeferredCleanup(discardPooled || this.#completionFailure !== null);
    if (this.#completionFailure) throw this.#completionFailure.error;
  }

  async submitAndWait() {
    this.submit();
    if (this.#cleanupPromise) {
      await this.#cleanupPromise;
    } else {
      await this.device.queue.onSubmittedWorkDone();
      // Safe to destroy evicted uniform buffers now that GPU work is complete
      this.#uniformCache?.flushPendingDestruction();
    }
    if (this.#completionFailure) throw this.#completionFailure.error;
  }

  getStats() {
    const opLabelCounts = this.#opLabelCounts
      ? Object.fromEntries(
        Object.entries(this.#opLabelCounts).sort((a, b) => {
          const countDelta = b[1] - a[1];
          return countDelta !== 0 ? countDelta : a[0].localeCompare(b[0]);
        })
      )
      : {};
    return {
      opCount: this.#opCount,
      opLabelCounts,
      dispatches: (this.#dispatchRecords ?? []).map((record) => ({
        ...record,
        workgroups: record.workgroups ? [...record.workgroups] : null,
      })),
      computePassCount: this.#computePassCount,
      profiling: {
        timingClass: this.#profileRequested ? 'instrumented-diagnostic' : 'ordinary-request',
        enabled: this.#profilingEnabled,
        phase: this.#profilePhase,
        timestampedPassCount: this.#nextQueryIndex / 2,
        untimestampedPassCount: this.#computePassCount - this.#nextQueryIndex / 2,
      },
      ...(this.#profileRequested ? { profile: structuredClone({
        status: this.#profilingEnabled ? 'pending' : 'unavailable',
        recorderLabel: this.label, computePasses: this.#computePassCount,
        measuredPasses: 0, intervals: [], ...this.#profileReport,
        unmeasuredPasses: this.#computePassCount - (this.#profileReport?.measuredPasses ?? 0),
        scope: 'Recorded compute passes only; copies and other queues are unaccounted for.',
      }) } : {}),
      tempBufferCount: this.#tempBuffers.length,
      pooledBufferCount: this.#pooledBuffers.length,
      submitted: this.#submitted,
      submissionCount: this.#submissionCount,
    };
  }

  getSubmitLatencyMs() {
    return this.#submitLatencyMs;
  }

  abort() {
    if (this.#submitted) {
      if (!this.#querySet) return this.#cleanupPromise;
      // A queue failure can prevent timing resolution after submit. Query
      // resources remain owned until completion or device-loss notification.
      this.#cleanupPromise = Promise.allSettled([
        this.#cleanupPromise, this.device.queue.onSubmittedWorkDone(),
      ]).then(() => this.#destroyProfilingResources());
      return this.#cleanupPromise;
    }

    this.closeActiveComputePass();

    const { buffersToDestroy, buffersToRelease } = this.#takeTrackedBuffers();
    this.#submitted = true;
    this.#cleanupAbortedBuffers(buffersToDestroy, buffersToRelease);
    return this.#cleanupPromise;
  }

  async resolveProfileTimings(options = {}) {
    this.#profileReport ??= { status: 'unavailable', measuredPasses: 0, intervals: [] };
    if (!this.#profilingEnabled || !this.#querySet || !this.#queryBuffer || !this.#readbackBuffer) {
      if (options.requireComplete) throw new Error('[CommandRecorder] Complete timestamp profiling is unavailable.');
      return null;
    }

    if (!this.#submitted) {
      throw new Error('[CommandRecorder] Must submit before resolving timings');
    }

    if (this.#profileEntries.length === 0) {
      try {
        await this.completeDeferredCleanup();
        this.#profileReport.status = this.#computePassCount === 0 ? 'complete' : 'incomplete';
        if (options.requireComplete && this.#computePassCount !== 0) {
          throw new Error('[CommandRecorder] Timestamp query capacity did not cover every compute pass.');
        }
      } finally {
        this.#destroyProfilingResources();
      }
      return {};
    }

    let mapped = false;

    try {
      this.#profileReport.status = 'failed';
      await this.device.queue.onSubmittedWorkDone();
      await this.completeDeferredCleanup();
      if (options.requireComplete && this.#profileEntries.length !== this.#computePassCount) {
        throw new Error('[CommandRecorder] Timestamp query capacity did not cover every compute pass.');
      }

      const maxIndex = Math.max(...this.#profileEntries.map(e => e.endQueryIndex)) + 1;
      const resolveEncoder = this.device.createCommandEncoder({ label: 'profile_resolve' });
      resolveEncoder.resolveQuerySet(this.#querySet, 0, maxIndex, this.#queryBuffer, 0);
      resolveEncoder.copyBufferToBuffer(this.#queryBuffer, 0, this.#readbackBuffer, 0, maxIndex * 8);
      this.device.queue.submit([resolveEncoder.finish()]);

      if (!allowReadback('CommandRecorder.resolveProfileTimings')) {
        if (options.requireComplete) throw new Error('[CommandRecorder] Timestamp readback is disabled.');
        return null;
      }

      await this.#readbackBuffer.mapAsync(GPUMapMode.READ);
      mapped = true;
      const timestamps = new BigUint64Array(this.#readbackBuffer.getMappedRange());
      const timings = {};

      for (const entry of this.#profileEntries) {
        const startNs = timestamps[entry.startQueryIndex];
        const endNs = timestamps[entry.endQueryIndex];
        const durationMs = Number(endNs - startNs) / 1_000_000;

        if (options.requireComplete && durationMs < 0) {
          throw new Error('[CommandRecorder] GPU timestamp order is invalid.');
        }
        if (!options.requireComplete && (durationMs < 0 || durationMs > 60000)) {
          continue;
        }

        this.#profileReport.intervals.push({ label: entry.label, dispatch: entry.dispatch,
          count: entry.dispatch ? 1 : null, durationMs });
        this.#profileReport.measuredPasses++;
        const group = options.groupByPhase === true
          ? (timings[entry.phase ?? 'unattributed'] ??= {})
          : timings;
        group[entry.label] = (group[entry.label] ?? 0) + durationMs;
      }

      this.#profileReport.status = this.#profileReport.measuredPasses === this.#computePassCount ? 'complete' : 'incomplete';
      return timings;
    } finally {
      if (mapped && this.#readbackBuffer) {
        this.#readbackBuffer.unmap();
      }
      this.#destroyProfilingResources();
    }
  }

  static formatProfileReport(timings) {
    const entries = Object.entries(timings).sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((sum, [, t]) => sum + t, 0);

    let report = 'GPU Profile Report\n';
    report += '\u2500'.repeat(50) + '\n';
    report += 'Kernel'.padEnd(25) + 'Time (ms)'.padStart(12) + '%'.padStart(8) + '\n';
    report += '\u2500'.repeat(50) + '\n';

    for (const [label, time] of entries) {
      const pct = (time / total * 100).toFixed(1);
      report += label.padEnd(25) + time.toFixed(2).padStart(12) + pct.padStart(8) + '\n';
    }

    report += '\u2500'.repeat(50) + '\n';
    report += 'TOTAL'.padEnd(25) + total.toFixed(2).padStart(12) + '100.0'.padStart(8) + '\n';

    return report;
  }

  #destroyProfilingResources() {
    if (this.#querySet) {
      this.#querySet.destroy();
      this.#querySet = null;
    }
    if (this.#queryBuffer) {
      this.#queryBuffer.destroy();
      this.#queryBuffer = null;
    }
    if (this.#readbackBuffer) {
      this.#readbackBuffer.destroy();
      this.#readbackBuffer = null;
    }
    this.#profileEntries = [];
  }
}

export function createCommandRecorder(label = 'command_recorder', options, device = null) {
  return new CommandRecorder(device, label, options);
}

export function createProfilingRecorder(label = 'profiled_recorder', device = null, options = {}) {
  return new CommandRecorder(device, label, { ...options, profile: true });
}

export default CommandRecorder;
