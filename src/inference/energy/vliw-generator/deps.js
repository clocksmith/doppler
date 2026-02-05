import { VLEN as VLIW_VLEN } from '../vliw-shared.js';

function vecAddrs(base) {
  return Array.from({ length: VLIW_VLEN }, (_, i) => base + i);
}

function readsWrites(op) {
  const { engine, slot } = op;
  if (engine === 'alu') {
    const [, dest, a1, a2] = slot;
    return { reads: [a1, a2], writes: [dest] };
  }
  if (engine === 'load') {
    switch (slot[0]) {
      case 'const':
        return { reads: [], writes: [slot[1]] };
      case 'load':
        return { reads: [slot[2]], writes: [slot[1]] };
      case 'vload':
        return { reads: [slot[2]], writes: vecAddrs(slot[1]) };
      case 'load_offset':
        return {
          reads: [slot[2], ...vecAddrs(slot[3])],
          writes: vecAddrs(slot[1]),
        };
      default:
        throw new Error(`Unknown load op ${slot[0]}`);
    }
  }
  if (engine === 'store') {
    switch (slot[0]) {
      case 'store':
        return { reads: [slot[1], slot[2]], writes: [] };
      case 'vstore':
        return { reads: [slot[1], ...vecAddrs(slot[2])], writes: [] };
      default:
        throw new Error(`Unknown store op ${slot[0]}`);
    }
  }
  if (engine === 'flow') {
    switch (slot[0]) {
      case 'add_imm':
        return { reads: [slot[2]], writes: [slot[1]] };
      case 'select':
        return { reads: [slot[2], slot[3], slot[4]], writes: [slot[1]] };
      case 'vselect':
        return {
          reads: [...vecAddrs(slot[2]), ...vecAddrs(slot[3]), ...vecAddrs(slot[4])],
          writes: vecAddrs(slot[1]),
        };
      default:
        throw new Error(`Unknown flow op ${slot[0]}`);
    }
  }
  if (engine === 'valu') {
    switch (slot[0]) {
      case 'vbroadcast':
        return { reads: [slot[2]], writes: vecAddrs(slot[1]) };
      case 'multiply_add':
        return {
          reads: [...vecAddrs(slot[2]), ...vecAddrs(slot[3]), ...vecAddrs(slot[4])],
          writes: vecAddrs(slot[1]),
        };
      default:
        return { reads: [...vecAddrs(slot[2]), ...vecAddrs(slot[3])], writes: vecAddrs(slot[1]) };
    }
  }
  if (engine === 'debug') {
    return { reads: [], writes: [] };
  }
  throw new Error(`Unknown engine ${engine}`);
}

export function buildDeps(ops, options = {}) {
  const includeTemp = options?.includeTemp !== false;
  const readsList = [];
  const writesList = [];
  ops.forEach((op) => {
    const { reads, writes } = readsWrites(op);
    readsList.push(reads);
    writesList.push(writes);
  });

  const deps = Array.from({ length: ops.length }, () => []);
  const depsLatency = Array.from({ length: ops.length }, () => []);
  const lastWrite = new Map();
  const lastRead = new Map();
  const lastTemp = new Map();

  for (let i = 0; i < ops.length; i++) {
    const reads = readsList[i];
    const writes = writesList[i];
    reads.forEach((addr) => {
      if (lastWrite.has(addr)) {
        const dep = lastWrite.get(addr);
        deps[i].push(dep);
        depsLatency[i].push([dep, 1]);
      }
    });
    writes.forEach((addr) => {
      if (lastWrite.has(addr)) {
        const dep = lastWrite.get(addr);
        deps[i].push(dep);
        depsLatency[i].push([dep, 1]);
      }
      if (lastRead.has(addr)) {
        const dep = lastRead.get(addr);
        deps[i].push(dep);
        depsLatency[i].push([dep, 0]);
      }
    });
    let temps = [];
    const tempMeta = ops[i].meta?.temp;
    if (tempMeta) {
      temps = typeof tempMeta === 'string' ? [tempMeta] : Array.from(tempMeta);
    }
    if (includeTemp) {
      temps.forEach((key) => {
        if (lastTemp.has(key)) {
          const dep = lastTemp.get(key);
          deps[i].push(dep);
          depsLatency[i].push([dep, 1]);
        }
      });
    }

    reads.forEach((addr) => lastRead.set(addr, i));
    writes.forEach((addr) => {
      lastWrite.set(addr, i);
      lastRead.delete(addr);
    });
    if (includeTemp) {
      temps.forEach((key) => lastTemp.set(key, i));
    }
  }

  return { deps, depsLatency, readsList, writesList };
}
