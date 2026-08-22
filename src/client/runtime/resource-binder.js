/** Generic TargetPlan slot -> physical WebGPU resource binding. */

function resolveGpuDevice(devicePort) {
  const device = typeof devicePort?.getDevice === 'function' ? devicePort.getDevice() : devicePort?.gpuDevice ?? devicePort;
  if (!device || typeof device.createBuffer !== 'function') {
    throw new Error('ResourceBinder requires a physical GPUDevice with createBuffer().');
  }
  return device;
}

function align(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

/** @param {object} expression @param {Record<string, number>} dimensions */
export function evaluateMemoryExpression(expression, dimensions) {
  if (expression?.op === 'constant') return expression.bytes;
  if (expression?.op !== 'affine') throw new Error(`Unsupported memory expression op "${expression?.op}".`);
  let bytes = expression.constantBytes;
  for (const [dimension, coefficient] of Object.entries(expression.terms)) {
    const value = dimensions[dimension];
    if (!Number.isInteger(value) || value < 0) throw new Error(`Memory expression requires non-negative integer dimension "${dimension}".`);
    bytes += value * coefficient;
  }
  return Math.max(expression.minimumBytes, align(bytes, expression.alignment));
}

function usageBit(name) {
  const usage = globalThis.GPUBufferUsage;
  if (!usage) throw new Error('ResourceBinder requires GPUBufferUsage globals.');
  const bits = {
    'copy-src': usage.COPY_SRC,
    'copy-dst': usage.COPY_DST,
    index: usage.INDEX,
    indirect: usage.INDIRECT,
    query: usage.QUERY_RESOLVE,
    storage: usage.STORAGE,
    uniform: usage.UNIFORM,
    vertex: usage.VERTEX,
  };
  if (!bits[name]) throw new Error(`ResourceBinder does not recognize GPU buffer usage "${name}".`);
  return bits[name];
}

function resolveUsage(slot) {
  if (Number.isInteger(slot.usageBits) && slot.usageBits > 0) return slot.usageBits;
  if (!Array.isArray(slot.usage) || slot.usage.length === 0) throw new Error(`TargetPlan slot "${slot.slotId}" has no GPU usage.`);
  return slot.usage.reduce((mask, name) => mask | usageBit(name), 0);
}

/** @param {object} devicePort @param {{ bindProgramSlot?: Function }} [program] */
export function createResourceBinder(devicePort, program = null) {
  const device = resolveGpuDevice(devicePort);
  const boundSlots = new Map();

  function destroyRecord(record) {
    if (record.owner === 'runtime') record.buffer?.destroy?.();
    if (record.owner === 'program') program?.releaseProgramSlot?.(record.slotId, record.resource);
  }

  return {
    bindSlots(memoryLayout, dynamicDimensions = {}) {
      if (!memoryLayout || !Array.isArray(memoryLayout.bufferSlots)) {
        throw new Error('bindSlots requires TargetPlan.memoryLayout.bufferSlots.');
      }
      for (const slot of memoryLayout.bufferSlots) {
        const sizeBytes = evaluateMemoryExpression(slot.size, dynamicDimensions);
        const existing = boundSlots.get(slot.slotId);
        if (existing && existing.sizeBytes === sizeBytes) continue;
        if (existing) destroyRecord(existing);
        if (slot.owner === 'program') {
          const resource = program?.bindProgramSlot?.(slot, sizeBytes, dynamicDimensions) ?? null;
          boundSlots.set(slot.slotId, { ...slot, sizeBytes, dimensions: { ...dynamicDimensions }, resource });
          continue;
        }
        const buffer = device.createBuffer({
          label: `doppler-pack:${slot.slotId}`,
          size: sizeBytes,
          usage: resolveUsage(slot),
        });
        boundSlots.set(slot.slotId, { ...slot, sizeBytes, dimensions: { ...dynamicDimensions }, buffer });
      }
      return boundSlots;
    },

    writeSlot(slotId, data, offset = 0) {
      const record = boundSlots.get(slotId);
      if (!record?.buffer) throw new Error(`Runtime-owned GPU slot "${slotId}" is not bound.`);
      const view = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
      device.queue.writeBuffer(record.buffer, offset, view.buffer, view.byteOffset, view.byteLength);
    },

    getSlot(slotId) {
      return boundSlots.get(slotId);
    },

    releaseTransient() {
      for (const [slotId, record] of boundSlots) {
        if (record.scope === 'transient' || record.scope === 'layer-recycled') {
          destroyRecord(record);
          boundSlots.delete(slotId);
        }
      }
    },

    releaseAll() {
      for (const record of boundSlots.values()) destroyRecord(record);
      boundSlots.clear();
    },
  };
}
