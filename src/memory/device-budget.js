// Accounts every GPUBuffer created on a Doppler device, including allocations
// outside the reusable pool. This is an application allocation ceiling, not
// physical VRAM/RSS accounting or a promise of immediate driver reclamation.
/** @typedef {{bytes: number, label: string, category: 'weights'|'attention'|'recurrent'|'other'}} Allocation */
/** @typedef {{maxBytes: number|null, liveBytes: number, peakBytes: number, allocations: number,
 * rejected: number, buffers: Map<GPUBuffer, Allocation>, lost: boolean}} Budget */
/** @type {WeakMap<GPUDevice, Budget>} */
const budgets = new WeakMap();

/** @type {import('./device-budget.js').installDeviceMemoryAccounting} */
export function installDeviceMemoryAccounting(device) {
  if (budgets.has(device)) return;
  /** @type {Budget} */
  const state = { maxBytes: null, liveBytes: 0, peakBytes: 0, allocations: 0,
    rejected: 0, buffers: new Map(), lost: false };
  budgets.set(device, state);
  const create = device.createBuffer.bind(device);
  device.createBuffer = descriptor => {
    const size = Number(descriptor.size);
    if (!Number.isSafeInteger(size) || size < 0) throw new Error('GPU allocation size must be a nonnegative safe integer.');
    if (state.lost) throw new Error('Cannot allocate on a lost GPU device.');
    if (state.maxBytes !== null && size > state.maxBytes - state.liveBytes) {
      state.rejected++;
      throw new Error(`GPU memory budget exceeded: ${descriptor.label || 'unlabeled'} requires ${size} bytes; `
        + `${state.liveBytes} live, ${state.maxBytes} allowed.`);
    }
    const buffer = create(descriptor);
    /** @type {Allocation} */
    const record = { bytes: size, label: String(descriptor.label || ''), category: 'other' };
    if (/^(kv_cache_|qkv_cache_|mixed_kv_)/.test(record.label)) record.category = 'attention';
    else if (/\.linear_(conv|recurrent)_state/.test(record.label)) record.category = 'recurrent';
    state.buffers.set(buffer, record);
    state.liveBytes += size;
    state.peakBytes = Math.max(state.peakBytes, state.liveBytes);
    state.allocations++;
    const destroy = buffer.destroy.bind(buffer);
    buffer.destroy = () => {
      // Keep the allocation charged if destruction itself fails.
      destroy();
      if (state.buffers.delete(buffer)) state.liveBytes -= size;
    };
    return buffer;
  };
  const lost = () => { state.lost = true; state.buffers.clear(); state.liveBytes = 0; };
  device.lost?.then(lost, lost);
}

/** @type {import('./device-budget.js').setDeviceMemoryBudget} */
export function setDeviceMemoryBudget(device, maxBytes) {
  if (maxBytes !== null && (!Number.isSafeInteger(maxBytes) || maxBytes <= 0)) {
    throw new Error('Device memory budget requires positive maxBytes or explicit null.');
  }
  installDeviceMemoryAccounting(device);
  const state = budgets.get(device);
  if (!state) throw new Error('Device allocation accounting was not installed.');
  if (state.liveBytes > 0 && state.maxBytes !== null && state.maxBytes !== maxBytes) {
    throw new Error('Release existing device allocations before changing their memory budget.');
  }
  if (maxBytes !== null && state.liveBytes > maxBytes) throw new Error('Existing GPU allocations exceed the requested memory budget.');
  state.maxBytes = maxBytes;
  return /** @type {import('./device-budget.js').DeviceMemorySnapshot} */ (getDeviceMemorySnapshot(device));
}

/** @type {import('./device-budget.js').markDeviceWeights} */
export function markDeviceWeights(device, buffers) {
  const state = budgets.get(device);
  for (const buffer of buffers) {
    const record = state?.buffers.get(buffer);
    if (record) record.category = 'weights';
  }
}

/** @type {import('./device-budget.js').getDeviceMemorySnapshot} */
export function getDeviceMemorySnapshot(device) {
  if (!device) return null;
  const state = budgets.get(device);
  if (!state) return null;
  const categories = { weights: 0, attention: 0, recurrent: 0, other: 0 };
  const labels = new Map();
  for (const record of state.buffers.values()) {
    categories[record.category] += record.bytes;
    const row = labels.get(record.label) || { label: record.label, category: record.category, bytes: 0, count: 0 };
    row.bytes += record.bytes; row.count++;
    labels.set(record.label, row);
  }
  return { scope: 'device-gpu-buffers', maxBytes: state.maxBytes, liveBytes: state.liveBytes,
    peakBytes: state.peakBytes, allocations: state.allocations, rejected: state.rejected,
    categories, labels: [...labels.values()] };
}
