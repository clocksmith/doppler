import { clearDequantCache, getDequantCacheStats, setDequantCacheMaxEntries } from './moe-cache.js';
import { moeFeedForwardCPU } from './moe-cpu.js';
import { moeFeedForwardGPU } from './moe-gpu.js';

export { clearDequantCache, getDequantCacheStats, setDequantCacheMaxEntries, moeFeedForwardCPU, moeFeedForwardGPU };

export function isMoELayer(_layerIdx) {
  return true;
}
