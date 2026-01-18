

import { getMemoryCapabilities } from '../memory/capability.js';
import { getQuotaInfo, formatBytes } from './quota.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { getStorageCapabilities } from './shard-manager.js';

// ============================================================================
// Model Requirements Constants
// ============================================================================

const GB = 1024 * 1024 * 1024;
const MB = 1024 * 1024;


export const GEMMA_1B_REQUIREMENTS = {
  modelId: 'gemma-3-1b-it-q4',
  displayName: 'Gemma 3 1B IT (Q4)',
  downloadSize: 537 * MB,       // ~537MB actual size from manifest shards
  vramRequired: 1.5 * GB,       // ~1.5GB VRAM for inference (weights + KV cache)
  paramCount: '1B',
  quantization: 'Q4_K_M',
  architecture: 'Gemma3ForCausalLM',
};


export const MODEL_REQUIREMENTS = {
  'gemma-3-1b-it-q4': GEMMA_1B_REQUIREMENTS,
};

// ============================================================================
// VRAM Estimation
// ============================================================================


function estimateAvailableVRAM(memCaps) {
  const info = memCaps.unifiedMemoryInfo;

  // Unified memory: use estimated system memory (leave headroom)
  if (info.isUnified && info.estimatedMemoryGB) {
    // Use configured ratio of unified memory available for GPU
    return (info.estimatedMemoryGB * GB) * getRuntimeConfig().loading.storage.vramEstimation.unifiedMemoryRatio;
  }

  // Discrete GPU: use maxBufferSize as heuristic
  // This isn't actual VRAM but gives us a sense of GPU capability
  if (info.limits?.maxBufferSize) {
    // Conservative: discrete GPUs can usually allocate ~80% of VRAM
    return info.limits.maxBufferSize;
  }

  // Fallback: use configured fallback VRAM
  return getRuntimeConfig().loading.storage.vramEstimation.fallbackVramBytes;
}


async function checkVRAM(
  requirements,
  memCaps
) {
  const available = estimateAvailableVRAM(memCaps);
  const required = requirements.vramRequired;
  const sufficient = available >= required;

  
  let message;
  if (sufficient) {
    message = `VRAM OK: ${formatBytes(available)} available, ${formatBytes(required)} required`;
  } else {
    message = `Insufficient VRAM: ${formatBytes(available)} available, ${formatBytes(required)} required`;
  }

  return { required, available, sufficient, message };
}

// ============================================================================
// Storage Check
// ============================================================================


async function checkStorage(
  requirements
) {
  const quotaInfo = await getQuotaInfo();
  const available = quotaInfo.available;
  const required = requirements.downloadSize;
  const sufficient = available >= required;

  
  let message;
  if (sufficient) {
    message = `Storage OK: ${formatBytes(available)} available, ${formatBytes(required)} required`;
  } else {
    const shortfall = required - available;
    message = `Insufficient storage: need ${formatBytes(shortfall)} more space`;
  }

  return { required, available, sufficient, message };
}

function resolveStorageBackend() {
  const config = getRuntimeConfig().loading.storage.backend;
  const caps = getStorageCapabilities();
  const requested = config.backend;
  let selected = 'memory';
  let persistent = false;
  let ok = true;
  let message = '';

  if (requested === 'opfs') {
    selected = 'opfs';
    persistent = true;
    ok = caps.opfs;
    message = caps.opfs ? 'OPFS available' : 'OPFS requested but not available';
  } else if (requested === 'indexeddb') {
    selected = 'indexeddb';
    persistent = true;
    ok = caps.indexeddb;
    message = caps.indexeddb ? 'IndexedDB available' : 'IndexedDB requested but not available';
  } else if (requested === 'memory') {
    selected = 'memory';
    persistent = false;
    message = 'Memory store requested';
  } else {
    if (caps.opfs) {
      selected = 'opfs';
      persistent = true;
      message = 'Auto-selected OPFS';
    } else if (caps.indexeddb) {
      selected = 'indexeddb';
      persistent = true;
      message = 'Auto-selected IndexedDB';
    } else {
      selected = 'memory';
      persistent = false;
      message = 'Auto-selected memory fallback';
    }
  }

  return {
    requested,
    selected,
    persistent,
    ok,
    message,
    caps,
  };
}

// ============================================================================
// GPU Check
// ============================================================================


async function checkGPU(memCaps) {
  const hasWebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;

  if (!hasWebGPU) {
    return {
      hasWebGPU: false,
      hasF16: false,
      device: 'WebGPU not available',
      isUnified: false,
    };
  }

  const info = memCaps.unifiedMemoryInfo;
  let device = 'Unknown GPU';

  if (info.apple?.isApple) {
    device = info.apple.description || `Apple M${info.apple.mSeriesGen || '?'}`;
  } else if (info.amd?.isAMDUnified) {
    device = info.amd.description || 'AMD Strix';
  } else if (info.limits?.maxBufferSize) {
    device = `GPU (${formatBytes(info.limits.maxBufferSize)} max buffer)`;
  }

  // Check for F16 support (need to request adapter to check features)
  let hasF16 = false;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter) {
      hasF16 = adapter.features.has('shader-f16');
    }
  } catch {
    // Ignore - hasF16 stays false
  }

  return {
    hasWebGPU: true,
    hasF16,
    device,
    isUnified: info.isUnified,
  };
}

// ============================================================================
// Public API
// ============================================================================


export async function runPreflightChecks(
  requirements
) {
  
  const warnings = [];
  
  const blockers = [];

  // Get memory capabilities (cached internally)
  const memCaps = await getMemoryCapabilities();

  // Run all checks
  const [vram, storage, gpu] = await Promise.all([
    checkVRAM(requirements, memCaps),
    checkStorage(requirements),
    checkGPU(memCaps),
  ]);
  const storageBackend = resolveStorageBackend();

  // Determine blockers
  if (!gpu.hasWebGPU) {
    blockers.push('WebGPU is not available in this browser');
  }

  if (!vram.sufficient) {
    blockers.push(vram.message);
  }

  if (!storage.sufficient) {
    blockers.push(storage.message);
  }
  if (!storageBackend.ok) {
    blockers.push(storageBackend.message);
  }

  // Determine warnings
  if (!gpu.hasF16) {
    warnings.push('F16 not supported - inference may be slower');
  }
  if (storageBackend.selected === 'indexeddb' && storageBackend.caps.opfs === false) {
    warnings.push('OPFS unavailable - using IndexedDB fallback (slower)');
  }
  if (storageBackend.selected === 'memory') {
    warnings.push('Persistent storage unavailable - using memory fallback');
    const maxBytes = getRuntimeConfig().loading.storage.backend.memory.maxBytes;
    if (maxBytes < requirements.downloadSize) {
      blockers.push('Memory fallback too small for model download');
    }
  }

  if (!gpu.isUnified && vram.sufficient) {
    // Discrete GPU with borderline VRAM
    const headroom = vram.available - vram.required;
    if (headroom < getRuntimeConfig().loading.storage.vramEstimation.lowVramHeadroomBytes) {
      warnings.push('Low VRAM headroom - may cause issues with longer contexts');
    }
  }

  const canProceed = blockers.length === 0;

  return {
    canProceed,
    vram,
    storage,
    gpu,
    storageBackend,
    warnings,
    blockers,
  };
}


export function formatPreflightResult(result) {
  
  const lines = [];

  lines.push(`GPU: ${result.gpu.device}`);
  lines.push(`VRAM: ${result.vram.message}`);
  lines.push(`Storage: ${result.storage.message}`);
  if (result.storageBackend) {
    lines.push(`Storage backend: ${result.storageBackend.selected} (${result.storageBackend.message})`);
  }

  if (result.warnings.length > 0) {
    lines.push(`Warnings: ${result.warnings.join('; ')}`);
  }

  if (result.blockers.length > 0) {
    lines.push(`Blockers: ${result.blockers.join('; ')}`);
  }

  lines.push(`Can proceed: ${result.canProceed ? 'Yes' : 'No'}`);

  return lines.join('\n');
}
