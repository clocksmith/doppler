import { Worker } from 'node:worker_threads';

function normalizeError(error) {
  if (error instanceof Error) return error;
  return new Error(String(error));
}

function buildWorkerError(payload) {
  const message = payload?.message ?? 'Worker transform failed';
  const error = new Error(message);
  if (payload?.stack) {
    error.stack = payload.stack;
  }
  return error;
}

export class NodeConvertWorkerPool {
  #states = [];
  #queue = [];
  #jobs = new Map();
  #nextJobId = 1;
  #destroyed = false;
  #failure = null;

  constructor(options = {}) {
    const size = options.size;
    if (!Number.isInteger(size) || size < 1) {
      throw new Error('node convert worker pool size must be a positive integer.');
    }

    const workerUrl = new URL('./node-convert-worker.js', import.meta.url);
    for (let i = 0; i < size; i++) {
      const worker = new Worker(workerUrl, { type: 'module' });
      const state = { worker, busy: false, activeJobId: null };
      worker.on('message', (message) => this.#onWorkerMessage(state, message));
      worker.on('error', (error) => this.#onWorkerFailure(state, error));
      worker.on('exit', (code) => {
        if (code !== 0 && !this.#destroyed) {
          this.#onWorkerFailure(
            state,
            new Error(`node convert worker exited with code ${code}`)
          );
        }
      });
      this.#states.push(state);
    }
  }

  get size() {
    return this.#states.length;
  }

  async transformTensor(tensor, tensorData, transformContext) {
    if (this.#destroyed) {
      throw this.#failure ?? new Error('node convert worker pool is closed.');
    }
    if (!tensor || typeof tensor !== 'object') {
      throw new Error('node convert worker job tensor is required.');
    }
    if (!(tensorData instanceof Uint8Array)) {
      throw new Error('node convert worker job tensorData must be Uint8Array.');
    }

    const bytes = tensorData.byteOffset === 0 && tensorData.byteLength === tensorData.buffer.byteLength
      ? tensorData
      : tensorData.slice();

    return this.#enqueueJob({
      tensor,
      tensorData: bytes.buffer,
      transformContext: transformContext ?? {},
    }, [bytes.buffer]);
  }

  async close() {
    if (this.#destroyed) return;
    this.#destroyed = true;
    const queuedError = new Error('node convert worker pool closed');
    while (this.#queue.length > 0) {
      const queued = this.#queue.shift();
      queued.reject(queuedError);
    }
    for (const job of this.#jobs.values()) {
      job.reject(queuedError);
    }
    this.#jobs.clear();

    const workers = this.#states.map((state) => state.worker.terminate());
    await Promise.allSettled(workers);
    this.#states = [];
  }

  #enqueueJob(payload, transferList) {
    const id = this.#nextJobId++;
    return new Promise((resolve, reject) => {
      this.#queue.push({
        id,
        payload,
        transferList,
        resolve,
        reject,
      });
      this.#dispatch();
    });
  }

  #dispatch() {
    if (this.#destroyed || this.#failure) return;
    for (const state of this.#states) {
      if (state.busy) continue;
      const job = this.#queue.shift();
      if (!job) break;
      state.busy = true;
      state.activeJobId = job.id;
      this.#jobs.set(job.id, {
        ...job,
        state,
      });
      try {
        state.worker.postMessage({
          id: job.id,
          job: job.payload,
        }, job.transferList);
      } catch (error) {
        this.#jobs.delete(job.id);
        state.busy = false;
        state.activeJobId = null;
        job.reject(normalizeError(error));
      }
    }
  }

  #onWorkerMessage(state, message) {
    const id = message?.id;
    if (!Number.isInteger(id)) {
      this.#onWorkerFailure(state, new Error('node convert worker returned invalid message id'));
      return;
    }
    const job = this.#jobs.get(id);
    if (!job) {
      this.#onWorkerFailure(state, new Error(`node convert worker returned unknown job id ${id}`));
      return;
    }
    this.#jobs.delete(id);
    state.busy = false;
    state.activeJobId = null;

    if (message?.ok === true) {
      const resultBytes = message?.result?.tensorData;
      if (!(resultBytes instanceof ArrayBuffer)) {
        job.reject(new Error('node convert worker returned invalid tensorData.'));
      } else {
        job.resolve({
          tensorData: new Uint8Array(resultBytes),
          outDtype: message?.result?.outDtype ?? null,
          outLayout: message?.result?.outLayout ?? null,
        });
      }
      this.#dispatch();
      return;
    }

    job.reject(buildWorkerError(message?.error));
    this.#dispatch();
  }

  #onWorkerFailure(state, error) {
    const normalized = normalizeError(error);
    this.#failure = normalized;
    this.#destroyed = true;

    while (this.#queue.length > 0) {
      const queued = this.#queue.shift();
      queued.reject(normalized);
    }
    for (const [jobId, job] of this.#jobs.entries()) {
      job.reject(normalized);
      this.#jobs.delete(jobId);
    }
    for (const st of this.#states) {
      st.busy = false;
      st.activeJobId = null;
      st.worker.terminate().catch(() => {});
    }
  }
}
