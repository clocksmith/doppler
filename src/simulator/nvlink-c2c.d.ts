/**
 * NVLink-C2C (Chip-to-Chip) Module
 *
 * Simulates the coherent memory interface between Grace CPU
 * and Hopper/Blackwell GPU in GH200/GB200 superchips.
 *
 * @module simulator/nvlink-c2c
 */

import type { VirtualGPU, VirtualCPU, VirtualBufferRef } from './virtual-device.js';
import type { TimingModel } from './timing-model.js';
import type { NVLinkC2CSpec } from '../config/schema/emulation.schema.js';

// =============================================================================
// Transfer Types
// =============================================================================

/**
 * Direction of C2C transfer
 */
export type C2CTransferDirection = 'cpu_to_gpu' | 'gpu_to_cpu';

/**
 * Result of a C2C transfer operation
 */
export interface C2CTransferResult {
  /** Transfer ID */
  transferId: string;
  /** Bytes transferred */
  bytesTransferred: number;
  /** Direction of transfer */
  direction: C2CTransferDirection;
  /** Simulated transfer time in ms */
  simulatedTimeMs: number;
  /** Actual transfer time in ms */
  actualTimeMs: number;
  /** Source (CPU or GPU index) */
  source: { type: 'cpu' | 'gpu'; index: number };
  /** Destination (CPU or GPU index) */
  destination: { type: 'cpu' | 'gpu'; index: number };
}

// =============================================================================
// NVLink-C2C Controller
// =============================================================================

/**
 * NVLink-C2C controller for CPU↔GPU coherent memory
 */
export declare class NVLinkC2CController {
  /** C2C specification */
  readonly spec: NVLinkC2CSpec;

  /** Timing model for delay injection */
  readonly timingModel: TimingModel;

  constructor(spec: NVLinkC2CSpec, timingModel: TimingModel);

  /**
   * Transfer data from CPU memory to GPU VRAM
   * @param cpu - Source CPU
   * @param cpuBufferId - CPU buffer ID
   * @param gpu - Destination GPU
   * @param gpuBufferId - GPU buffer ID (created if not provided)
   * @param sizeBytes - Bytes to transfer
   */
  cpuToGpu(
    cpu: VirtualCPU,
    cpuBufferId: string,
    gpu: VirtualGPU,
    gpuBufferId?: string,
    sizeBytes?: number
  ): Promise<C2CTransferResult>;

  /**
   * Transfer data from GPU VRAM to CPU memory
   * @param gpu - Source GPU
   * @param gpuBufferId - GPU buffer ID
   * @param cpu - Destination CPU
   * @param cpuBufferId - CPU buffer ID (created if not provided)
   * @param sizeBytes - Bytes to transfer
   */
  gpuToCpu(
    gpu: VirtualGPU,
    gpuBufferId: string,
    cpu: VirtualCPU,
    cpuBufferId?: string,
    sizeBytes?: number
  ): Promise<C2CTransferResult>;

  /**
   * Perform coherent access (simulates unified memory)
   * GPU can access CPU memory directly with C2C latency
   * @param cpu - CPU containing the data
   * @param cpuBufferId - CPU buffer ID
   * @param offset - Offset in bytes
   * @param length - Length in bytes
   */
  coherentRead(
    cpu: VirtualCPU,
    cpuBufferId: string,
    offset?: number,
    length?: number
  ): Promise<{ data: ArrayBuffer; transferResult: C2CTransferResult }>;

  /**
   * Perform coherent write
   * GPU can write to CPU memory directly
   * @param cpu - Target CPU
   * @param cpuBufferId - CPU buffer ID
   * @param data - Data to write
   * @param offset - Offset in bytes
   */
  coherentWrite(
    cpu: VirtualCPU,
    cpuBufferId: string,
    data: ArrayBuffer,
    offset?: number
  ): Promise<C2CTransferResult>;

  /**
   * Get transfer statistics
   */
  getStats(): NVLinkC2CStats;

  /**
   * Reset statistics
   */
  resetStats(): void;
}

/**
 * Statistics for NVLink-C2C transfers
 */
export interface NVLinkC2CStats {
  /** Total transfers completed */
  totalTransfers: number;
  /** Total bytes transferred */
  totalBytesTransferred: number;
  /** Total simulated time in ms */
  totalSimulatedTimeMs: number;
  /** Total actual time in ms */
  totalActualTimeMs: number;
  /** CPU to GPU transfers */
  cpuToGpuTransfers: number;
  /** GPU to CPU transfers */
  gpuToCpuTransfers: number;
  /** Coherent reads */
  coherentReads: number;
  /** Coherent writes */
  coherentWrites: number;
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create an NVLink-C2C controller
 */
export declare function createNVLinkC2CController(
  spec: NVLinkC2CSpec,
  timingModel: TimingModel
): NVLinkC2CController;
