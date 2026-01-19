
import { log } from '../debug/index.js';
import { createEmulationConfig, formatBytes, formatBandwidth } from '../config/schema/emulation.schema.js';
import { createVirtualCluster, VirtualGPU, VirtualCPU, VirtualCluster } from './virtual-device.js';
import { createTimingModel, TimingModel } from './timing-model.js';
import { createNVLinkC2CController, NVLinkC2CController } from './nvlink-c2c.js';
import { createNVLinkFabric, NVLinkFabric } from './nvlink-fabric.js';
import { EmulatedVramStore, detectLocalResources } from '../storage/emulated-vram.js';

// =============================================================================
// Re-exports
// =============================================================================

export { VirtualGPU, VirtualCPU, VirtualCluster, createVirtualCluster } from './virtual-device.js';
export { TimingModel, createTimingModel, calculateTheoreticalFlops, flopsToTimeMs } from './timing-model.js';
export { NVLinkC2CController, createNVLinkC2CController } from './nvlink-c2c.js';
export { NVLinkFabric, createNVLinkFabric } from './nvlink-fabric.js';

// =============================================================================
// Constants
// =============================================================================

const MODULE = 'Simulator';

// =============================================================================
// Emulation Context Implementation
// =============================================================================

class EmulationContextImpl {
  constructor(config, cluster, timing, nvlinkC2C, nvlinkFabric, vramStore) {
    this.config = config;

    this.cluster = cluster;

    this.timing = timing;

    this.nvlinkC2C = nvlinkC2C;

    this.nvlinkFabric = nvlinkFabric;

    this.vramStore = vramStore;

    this.active = true;

    this._startTime = Date.now();
  }

  getGPU(index) {
    return this.cluster.getGPU(index);
  }

  getCPU(index) {
    return this.cluster.getCPU(index);
  }

  getStats() {
    const clusterStats = this.cluster.getClusterStats();
    const nvlinkStats = this.nvlinkFabric.getStats();
    const c2cStats = this.nvlinkC2C.getStats();
    const wallClockMs = Date.now() - this._startTime;

    return {
      gpuStats: clusterStats.gpuStats.map((gpuStat, i) => ({
        gpuIndex: i,
        vramAllocatedBytes: gpuStat.allocatedBytes,
        vramUsedBytes: gpuStat.vramUsedBytes,
        allocationCount: gpuStat.bufferCount,
        computeOps: 0, // TODO: Track compute ops
        computeTimeMs: 0,
      })),
      nvlinkStats: {
        totalBytesTransferred: nvlinkStats.totalBytesTransferred,
        transferCount: nvlinkStats.totalP2PTransfers + nvlinkStats.totalCollectives,
        simulatedTimeMs: nvlinkStats.totalSimulatedTimeMs,
        actualTimeMs: nvlinkStats.totalActualTimeMs,
      },
      nvlinkC2CStats: {
        totalBytesTransferred: c2cStats.totalBytesTransferred,
        transferCount: c2cStats.totalTransfers,
        simulatedTimeMs: c2cStats.totalSimulatedTimeMs,
        actualTimeMs: c2cStats.totalActualTimeMs,
      },
      totalInjectedDelayMs: this.timing.getTotalInjectedDelayMs(),
      wallClockTimeMs: wallClockMs,
      vramStore: this.vramStore ? this.vramStore.getStats() : undefined,
    };
  }

  resetStats() {
    this.timing.reset();
    this.nvlinkC2C.resetStats();
    this.nvlinkFabric.resetStats();
    this._startTime = Date.now();
  }

  async destroy() {
    if (!this.active) return;

    this.active = false;
    if (this.vramStore) {
      await this.vramStore.destroy();
    }
    await this.cluster.destroy();

    log.info(MODULE, 'Simulation context destroyed');
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

async function resolveLocalResourceBudgets(config) {
  if (config.localResources) {
    return {
      vramBytes: config.localResources.tier1Vram.maxBytes,
      ramBytes: config.localResources.tier2Ram.maxBytes,
      storageBytes: config.localResources.tier3Storage.maxBytes,
    };
  }

  return detectLocalResources();
}

function resolveVramBudgets(config, resources) {
  const vramBudgetBytes = Math.min(config.maxActiveWorkingSetBytes, resources.vramBytes);
  return {
    vramBudgetBytes,
    ramBudgetBytes: resources.ramBytes,
  };
}

export async function createEmulationContext(configOverrides) {
  // Merge with defaults
  const config = createEmulationConfig(configOverrides);

  if (!config.enabled) {
    throw new Error('Emulation is not enabled in config');
  }

  log.info(MODULE, `Creating simulation context for ${config.targetChip}`);
  log.info(MODULE, `  GPUs: ${config.topology.gpuCount} x ${config.gpuSpec.name} (${formatBytes(config.gpuSpec.vramBytes)} each)`);
  log.info(MODULE, `  CPUs: ${config.topology.cpuCount} x ${config.cpuSpec.name} (${formatBytes(config.cpuSpec.memoryBytes)} each)`);
  log.info(MODULE, `  NVLink: ${formatBandwidth(config.nvlink.bandwidthBytesPerSec)}`);
  log.info(MODULE, `  Timing mode: ${config.timingMode}`);

  const localResources = await resolveLocalResourceBudgets(config);
  const budgets = resolveVramBudgets(config, localResources);

  const vramStore = new EmulatedVramStore(
  config.opfsRootPath,
  budgets.vramBudgetBytes,
  budgets.ramBudgetBytes
  );
  await vramStore.initialize();

  for (let i = 0; i < config.topology.gpuCount; i++) {
    await vramStore.createPartition({
      name: `gpu${i}`,
      maxBytes: config.gpuSpec.vramBytes,
      opfsPath: `${config.opfsRootPath}/gpu${i}`,
    });
  }

  // Create components
  const cluster = createVirtualCluster(config);
  await cluster.initialize();

  const timing = createTimingModel(config);
  const nvlinkC2C = createNVLinkC2CController(config.nvlinkC2C, timing);
  const nvlinkFabric = createNVLinkFabric(
  config.nvlink,
  config.topology,
  timing,
  cluster
  );

  const ctx = new EmulationContextImpl(config, cluster, timing, nvlinkC2C, nvlinkFabric, vramStore);

  log.info(MODULE, `Simulation context ready: ${config.topology.gpuCount} virtual GPUs`);

  return ctx;
}

export async function createEmulationContextForChip(chipType, overrides = {}) {
  return createEmulationContext({
    enabled: true,
    targetChip: chipType,
    ...overrides,
  });
}

export async function isEmulationSupported() {
  if (typeof navigator === 'undefined') {
    log.warn(MODULE, 'Simulation requires browser APIs (navigator not available)');
    return false;
  }

  // Check WebGPU
  if (!navigator.gpu) {
    log.warn(MODULE, 'WebGPU not available');
    return false;
  }

  // Check OPFS
  try {
    const root = await navigator.storage.getDirectory();
    if (!root) {
      log.warn(MODULE, 'OPFS not available');
      return false;
    }
  } catch (err) {
    log.warn(MODULE, `OPFS check failed: ${err.message}`);
    return false;
  }

  return true;
}

export async function getEmulationCapabilities() {
  const result = {
    webgpuAvailable: false,
    opfsAvailable: false,
    estimatedVramBytes: 0,
    estimatedRamBytes: 0,
    estimatedStorageBytes: 0,
  };

  if (typeof navigator === 'undefined') {
    return result;
  }

  // Check WebGPU
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        result.webgpuAvailable = true;
      }
    } catch (err) {
      // WebGPU not available
    }
  }

  // Check OPFS
  try {
    await navigator.storage.getDirectory();
    result.opfsAvailable = true;
  } catch (err) {
    // OPFS not available
  }

  // Detect local resources
  try {
    const resources = await detectLocalResources();
    result.estimatedVramBytes = resources.vramBytes;
    result.estimatedRamBytes = resources.ramBytes;
    result.estimatedStorageBytes = resources.storageBytes;
  } catch (err) {
    log.warn(MODULE, `Resource detection failed: ${err.message}`);
  }

  return result;
}

// =============================================================================
// Default Export
// =============================================================================

export default {
  createEmulationContext,
  createEmulationContextForChip,
  isEmulationSupported,
  getEmulationCapabilities,
};
