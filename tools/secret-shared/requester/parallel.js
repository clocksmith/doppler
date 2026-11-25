/**
 * Worker threads for the requester's DeltaNet heads (Node worker_threads; a browser can do the same with Web Workers
 * when it is cross-origin isolated). `alloc(n)` returns SharedArrayBuffer-backed Float32Arrays so the state and the
 * per-layer inputs are visible to every worker without copies; `deltaRule(job)` splits the value heads.
 */
import os from 'node:os';
import { Worker, isMainThread, parentPort } from 'node:worker_threads';
import { deltaRuleHeads } from './delta-rule.js';

export function createHeadPool(threads = 0) {
  const count = threads > 0 ? threads : Math.min(8, os.availableParallelism());
  const workers = Array.from({ length: count }, () => new Worker(new URL(import.meta.url)));
  const waiting = new Map();
  let next = 1;
  for (const w of workers) {
    w.on('message', ({ id, error }) => { const p = waiting.get(id); waiting.delete(id); error ? p.reject(Error(error)) : p.resolve(); });
    w.unref();
  }
  return {
    threads: count,
    alloc: n => new Float32Array(new SharedArrayBuffer(4 * n)),
    async deltaRule({ S, q, k, v, g, beta, z, gnorm, out, n, d }) {
      const per = Math.ceil(d.VH / count), jobs = [];
      for (let i = 0; i * per < d.VH; i++) {
        const id = next++;
        jobs.push(new Promise((resolve, reject) => waiting.set(id, { resolve, reject })));
        workers[i].ref();
        workers[i].postMessage({ id, S, q, k, v, g, beta, z, gnorm, out, n, d, h0: i * per, h1: Math.min(d.VH, (i + 1) * per) });
      }
      try { await Promise.all(jobs); } finally { for (const w of workers) w.unref(); }
    },
    close() { for (const w of workers) w.terminate(); }
  };
}

if (!isMainThread) {
  parentPort.on('message', ({ id, S, q, k, v, g, beta, z, gnorm, out, n, d, h0, h1 }) => {
    try { deltaRuleHeads(S, q, k, v, g, beta, z, gnorm, out, n, d, h0, h1); parentPort.postMessage({ id }); }
    catch (error) { parentPort.postMessage({ id, error: String(error?.message ?? error) }); }
  });
}
