/**
 * NVIDIA Superchip Simulation Module
 *
 * Main entry point for the simulation layer. Provides factory functions
 * and an EmulationContext that manages all simulation components.
 *
 * @module simulator
 */

import type { EmulationConfigSchema, EmulationStats } from '../config/schema/emulation.schema.js';
import type { VirtualCluster, VirtualGPU, VirtualCPU } from './virtual-device.js';
import type { TimingModel } from './timing-model.js';
import type { NVLinkC2CController } from './nvlink-c2c.js';
import type { NVLinkFabric } from './nvlink-fabric.js';
import type { EmulatedVramStore } from '../storage/emulated-vram.js';

// Re-exports for convenience
export * from './virtual-device.js';
export * from './timing-model.js';
export * from './nvlink-c2c.js';
export * from './nvlink-fabric.js';
export type { EmulationConfigSchema, EmulationStats } from '../config/schema/emulation.schema.js';

// =============================================================================
// Emulation Context
// =============================================================================

/**
 * Emulation context holding all components for superchip emulation
 */
export interface EmulationContext {
  /** Emulation configuration */
  config: EmulationConfigSchema;

  /** Virtual cluster (GPUs + CPUs) */
  cluster: VirtualCluster;

  /** Timing model for delay injection */
  timing: TimingModel;

  /** NVLink-C2C controller (CPU↔GPU) */
  nvlinkC2C: NVLinkC2CController;

  /** NVLink fabric (GPU↔GPU) */
  nvlinkFabric: NVLinkFabric;

  /** Emulated VRAM store (tiered storage) */
  vramStore: EmulatedVramStore;

  /** Whether emulation is active */
  active: boolean;

  /**
   * Get the virtual GPU at specified index
   * @param index - GPU index
   */
  getGPU(index: number): VirtualGPU;

  /**
   * Get the virtual CPU at specified index
   * @param index - CPU index
   */
  getCPU(index: number): VirtualCPU;

  /**
   * Get comprehensive statistics
   */
  getStats(): EmulationStats;

  /**
   * Reset all statistics
   */
  resetStats(): void;

  /**
   * Destroy the emulation context and free resources
   */
  destroy(): Promise<void>;
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create an emulation context from configuration
 *
 * This is the main entry point for using the simulation layer.
 *
 * @param config - Emulation configuration (or partial with targetChip)
 * @returns Promise resolving to initialized EmulationContext
 *
 * @example
 * ```javascript
 * import { createEmulationContext } from 'doppler/simulator';
 *
 * const ctx = await createEmulationContext({
 *   enabled: true,
 *   targetChip: 'gh200',
 *   timingMode: 'functional',
 * });
 *
 * // Use the virtual cluster
 * const gpu0 = ctx.getGPU(0);
 * const buffer = await gpu0.allocate(1024 * 1024, 'test');
 *
 * // Check stats
 * console.log(ctx.getStats());
 *
 * // Clean up
 * await ctx.destroy();
 * ```
 */
export declare function createEmulationContext(
  config: Partial<EmulationConfigSchema>
): Promise<EmulationContext>;

/**
 * Create an emulation context for a specific chip preset
 *
 * Convenience function that loads the preset and creates the context.
 *
 * @param chipType - Target chip type ('gh200', 'gh200-nvl2', 'gb200-8gpu', 'gb200-nvl72')
 * @param overrides - Optional configuration overrides
 */
export declare function createEmulationContextForChip(
  chipType: string,
  overrides?: Partial<EmulationConfigSchema>
): Promise<EmulationContext>;

/**
 * Check if emulation is supported in the current environment
 *
 * Checks for required APIs: WebGPU, OPFS, etc.
 */
export declare function isEmulationSupported(): Promise<boolean>;

/**
 * Get a summary of emulation capabilities
 */
export declare function getEmulationCapabilities(): Promise<{
  webgpuAvailable: boolean;
  opfsAvailable: boolean;
  estimatedVramBytes: number;
  estimatedRamBytes: number;
  estimatedStorageBytes: number;
}>;
