import { buildDeps } from './deps.js';

class MinHeap {
  constructor(compare) {
    this.compare = compare;
    this.data = [];
  }

  push(item) {
    const data = this.data;
    data.push(item);
    let idx = data.length - 1;
    while (idx > 0) {
      const parent = Math.floor((idx - 1) / 2);
      if (this.compare(data[idx], data[parent]) >= 0) break;
      [data[idx], data[parent]] = [data[parent], data[idx]];
      idx = parent;
    }
  }

  pop() {
    const data = this.data;
    if (!data.length) return null;
    const root = data[0];
    const tail = data.pop();
    if (data.length && tail) {
      data[0] = tail;
      let idx = 0;
      while (true) {
        const left = idx * 2 + 1;
        const right = left + 1;
        let smallest = idx;
        if (left < data.length && this.compare(data[left], data[smallest]) < 0) {
          smallest = left;
        }
        if (right < data.length && this.compare(data[right], data[smallest]) < 0) {
          smallest = right;
        }
        if (smallest === idx) break;
        [data[idx], data[smallest]] = [data[smallest], data[idx]];
        idx = smallest;
      }
    }
    return root;
  }

  peek() {
    return this.data.length ? this.data[0] : null;
  }

  get size() {
    return this.data.length;
  }
}

function compareTuple(a, b) {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  if (a.length < b.length) return -1;
  if (a.length > b.length) return 1;
  return 0;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function scheduleOpsDepOnce(
  ops,
  caps,
  { returnOps = false, seed = 0, jitter = 0, useTempDeps = true } = {},
) {
  const { depsLatency, readsList, writesList } = buildDeps(ops, { includeTemp: useTempDeps });
  const nOps = ops.length;
  const children = Array.from({ length: nOps }, () => []);
  const indegree = Array.from({ length: nOps }, () => 0);

  for (let i = 0; i < nOps; i++) {
    depsLatency[i].forEach(([d, latency]) => {
      children[d].push([i, latency]);
      indegree[i] += 1;
    });
  }

  const priority = Array.from({ length: nOps }, () => 1);
  for (let i = nOps - 1; i >= 0; i--) {
    if (children[i].length) {
      let maxChild = 0;
      children[i].forEach(([c]) => {
        if (priority[c] > maxChild) maxChild = priority[c];
      });
      priority[i] = 1 + maxChild;
    }
  }

  const earliest = Array.from({ length: nOps }, () => 0);
  const scheduled = Array.from({ length: nOps }, () => -1);
  const ready = {};
  const rng = jitter > 0 ? mulberry32(seed) : null;

  const jitterKey = () => (rng ? rng() * jitter : 0);

  for (let i = 0; i < nOps; i++) {
    if (indegree[i] === 0) {
      const engine = ops[i].engine;
      if (!ready[engine]) {
        ready[engine] = new MinHeap(compareTuple);
      }
      ready[engine].push([0, -priority[i], jitterKey(), i]);
    }
  }

  const engineOrderBase = ['alu', 'valu', 'load', 'store', 'flow', 'debug'];
  const engineIndex = {};
  engineOrderBase.forEach((eng, idx) => {
    engineIndex[eng] = idx;
  });

  const instrs = [];
  let cycle = 0;
  let remaining = nOps;
  while (remaining > 0) {
    while (instrs.length <= cycle) {
      instrs.push({});
    }

    const writesCycle = new Set();
    const engineCounts = {};
    let anyScheduled = false;

    const releaseChildren = (idx) => {
      children[idx].forEach(([child, latency]) => {
        indegree[child] -= 1;
        earliest[child] = Math.max(earliest[child], scheduled[idx] + latency);
        if (indegree[child] === 0) {
          const childEngine = ops[child].engine;
          if (!ready[childEngine]) {
            ready[childEngine] = new MinHeap(compareTuple);
          }
          ready[childEngine].push([earliest[child], -priority[child], jitterKey(), child]);
        }
      });
    };

    let madeProgress = true;
    while (madeProgress) {
      madeProgress = false;
      const engineKey = (engine) => {
        const cap = caps?.[engine] ?? 0;
        if (cap <= 0 || (engineCounts[engine] || 0) >= cap) {
          return [0, -1, -engineIndex[engine]];
        }
        const heap = ready[engine];
        if (!heap || heap.size === 0) {
          return [0, -1, -engineIndex[engine]];
        }
        const top = heap.peek();
        if (top[0] > cycle) {
          return [0, -1, -engineIndex[engine]];
        }
        return [1, -top[1], -engineIndex[engine]];
      };

      const engineOrder = engineOrderBase.slice().sort((a, b) => {
        const ka = engineKey(a);
        const kb = engineKey(b);
        return compareTuple(kb, ka);
      });

      engineOrder.forEach((engine) => {
        const cap = caps?.[engine] ?? 0;
        if (cap <= 0) return;
        let count = engineCounts[engine] || 0;
        if (count >= cap) return;
        const heap = ready[engine];
        if (!heap || heap.size === 0) return;
        const skipped = [];
        while (heap.size && count < cap) {
          const [readyCycle, negPri, j, idx] = heap.pop();
          if (readyCycle > cycle) {
            skipped.push([readyCycle, negPri, j, idx]);
            break;
          }
          const writes = writesList[idx];
          if (writes.some((w) => writesCycle.has(w))) {
            skipped.push([readyCycle, negPri, j, idx]);
            continue;
          }
          const op = ops[idx];
          if (!instrs[cycle][engine]) instrs[cycle][engine] = [];
          instrs[cycle][engine].push(returnOps ? op : op.slot);
          scheduled[idx] = cycle;
          writes.forEach((w) => writesCycle.add(w));
          remaining -= 1;
          anyScheduled = true;
          madeProgress = true;
          count += 1;
          releaseChildren(idx);
        }
        skipped.forEach((item) => heap.push(item));
        engineCounts[engine] = count;
      });
    }

    if (!anyScheduled) {
      let nextCycle = null;
      Object.values(ready).forEach((heap) => {
        if (heap.size) {
          const rc = heap.peek()[0];
          if (nextCycle == null || rc < nextCycle) {
            nextCycle = rc;
          }
        }
      });
      if (nextCycle == null) break;
      cycle = Math.max(cycle + 1, nextCycle);
      continue;
    }

    cycle += 1;
  }

  return instrs;
}

export function scheduleOpsDep(
  ops,
  caps,
  { returnOps = false, seed = 0, jitter = 0, restarts = 1, useTempDeps = true } = {},
) {
  const safeRestarts = Math.max(1, Math.floor(restarts));
  if (safeRestarts === 1 || jitter <= 0) {
    return scheduleOpsDepOnce(ops, caps, { returnOps, seed, jitter, useTempDeps });
  }
  let bestInstrs = null;
  let bestCycles = null;
  for (let k = 0; k < safeRestarts; k++) {
    const instrs = scheduleOpsDepOnce(ops, caps, {
      returnOps,
      seed: seed + k,
      jitter,
      useTempDeps,
    });
    const cycles = countCycles(instrs);
    if (bestCycles == null || cycles < bestCycles) {
      bestCycles = cycles;
      bestInstrs = instrs;
    }
  }
  return bestInstrs || [];
}

export function countCycles(instrs) {
  let cycles = 0;
  instrs.forEach((bundle) => {
    for (const [engine, slots] of Object.entries(bundle)) {
      if (engine !== 'debug' && slots && slots.length) {
        cycles += 1;
        break;
      }
    }
  });
  return cycles;
}
