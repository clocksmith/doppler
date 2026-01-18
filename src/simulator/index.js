/**
 * NVIDIA Superchip Simulation Module
 *
 * Main entry point for the simulation layer. Provides factory functions
 * and an EmulationContext that manages all simulation components.
 *
 * @module simulator
 */

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

/**
 * Emulation context holding all components
 */
class EmulationContextImpl {
  /**
   * @param {import('../config/schema/emulation.schema.js').EmulationConfigSchema} config
   * @param {import('./virtual-device.js').VirtualCluster} cluster
   * @param {import('./timing-model.js').TimingModel} timing
   * @param {import('./nvlink-c2c.js').NVLinkC2CController} nvlinkC2C
   * @param {import('./nvlink-fabric.js').NVLinkFabric} nvlinkFabric
   */
  constructor(config, cluster, timing, nvlinkC2C, nvlinkFabric, vramStore) {
    /** @type {import('../config/schema/emulation.schema.js').EmulationConfigSchema} */
    this.config = config;

    /** @type {import('./virtual-device.js').VirtualCluster} */
    this.cluster = cluster;

    /** @type {import('./timing-model.js').TimingModel} */
    this.timing = timing;

    /** @type {import('./nvlink-c2c.js').NVLinkC2CController} */
    this.nvlinkC2C = nvlinkC2C;

    /** @type {import('./nvlink-fabric.js').NVLinkFabric} */
    this.nvlinkFabric = nvlinkFabric;

    this.vramStore = vramStore;

    /** @type {boolean} */
    this.active = true;

    /** @type {number} */
    this._startTime = Date.now();
  }

  /**
   * Get the virtual GPU at specified index
   * @param {number} index
   * @returns {import('./virtual-device.js').VirtualGPU}
   */
  getGPU(index) {
    return this.cluster.getGPU(index);
  }

  /**
   * Get the virtual CPU at specified index
   * @param {number} index
   * @returns {import('./virtual-device.js').VirtualCPU}
   */
  getCPU(index) {
    return this.cluster.getCPU(index);
  }

  /**
   * Get comprehensive statistics
   * @returns {import('../config/schema/emulation.schema.js').EmulationStats}
   */
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

  /**
   * Reset all statistics
   */
  resetStats() {
    this.timing.reset();
    this.nvlinkC2C.resetStats();
    this.nvlinkFabric.resetStats();
    this._startTime = Date.now();
  }

  /**
   * Destroy the emulation context
   */
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

/**
 * Create an emulation context from configuration
 * @param {Partial<import('../config/schema/emulation.schema.js').EmulationConfigSchema>} configOverrides
 * @returns {Promise<import('./index.js').EmulationContext>}
 */
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

/**
 * Create an emulation context for a specific chip preset
 * @param {string} chipType
 * @param {Partial<import('../config/schema/emulation.schema.js').EmulationConfigSchema>} [overrides]
 * @returns {Promise<import('./index.js').EmulationContext>}
 */
export async function createEmulationContextForChip(chipType, overrides = {}) {
  return createEmulationContext({
    enabled: true,
    targetChip: chipType,
    ...overrides,
  });
}

/**
 * Check if emulation is supported
 * @returns {Promise<boolean>}
 */
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

/**
 * Get emulation capabilities
 * @returns {Promise<{webgpuAvailable: boolean, opfsAvailable: boolean, estimatedVramBytes: number, estimatedRamBytes: number, estimatedStorageBytes: number}>}
 */
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
