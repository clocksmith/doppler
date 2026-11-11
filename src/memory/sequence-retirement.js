import { getBufferPool } from './buffer-pool.js';
import { getDeviceMemorySnapshot } from './device-budget.js';
import { isDeviceLost } from './device-residency.js';

// Under an explicit device memory budget, idle pooled buffers and pending
// retirements are charged as live allocations. A sequence boundary retires them
// after confirmed GPU completion, so one sequence's working set is bounded by its
// own allocations rather than by buckets left over from earlier prompt lengths.
// Without a device budget the pool keeps its reuse behavior. Owned (active)
// buffers are never touched.
/** @type {typeof import('./sequence-retirement.js').retireIdleBuffersUnderDeviceBudget} */
export async function retireIdleBuffersUnderDeviceBudget(device) {
  if (!device || isDeviceLost(device)) return false;
  if ((getDeviceMemorySnapshot(device)?.maxBytes ?? null) === null) return false;
  const pool = getBufferPool(device);
  pool.clearPool();
  await pool.settleDeferredDestruction();
  return true;
}
