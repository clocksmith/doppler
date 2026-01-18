/**
 * NVLink Fabric Module
 *
 * Simulates GPU-to-GPU communication over NVLink for tensor parallelism,
 * pipeline parallelism, and collective operations.
 *
 * @module simulator/nvlink-fabric
 */

import type { VirtualGPU, VirtualCluster, VirtualBufferRef } from './virtual-device.js';
import type { TimingModel } from './timing-model.js';
import type { NVLinkSpec, EmulatedClusterTopology } from '../config/schema/emulation.schema.js';

// =============================================================================
// Collective Operation Types
// =============================================================================

/**
 * Result of a collective operation
 */
export interface CollectiveResult {
  /** Operation ID */
  operationId: string;
  /** Type of collective */
  type: 'all_reduce' | 'all_gather' | 'reduce_scatter' | 'broadcast' | 'send' | 'recv';
  /** Participating GPU indices */
  gpuIndices: number[];
  /** Total bytes transferred */
  totalBytesTransferred: number;
  /** Simulated time in ms */
  simulatedTimeMs: number;
  /** Actual time in ms */
  actualTimeMs: number;
}

/**
 * Point-to-point transfer result
 */
export interface P2PTransferResult {
  /** Transfer ID */
  transferId: string;
  /** Source GPU index */
  srcGpu: number;
  /** Destination GPU index */
  dstGpu: number;
  /** Bytes transferred */
  bytesTransferred: number;
  /** Simulated time in ms */
  simulatedTimeMs: number;
  /** Actual time in ms */
  actualTimeMs: number;
}

// =============================================================================
// NVLink Fabric Controller
// =============================================================================

/**
 * NVLink fabric controller for GPU-to-GPU communication
 */
export declare class NVLinkFabric {
  /** NVLink specification */
  readonly spec: NVLinkSpec;

  /** Cluster topology */
  readonly topology: EmulatedClusterTopology;

  /** Timing model */
  readonly timingModel: TimingModel;

  constructor(
    spec: NVLinkSpec,
    topology: EmulatedClusterTopology,
    timingModel: TimingModel,
    cluster: VirtualCluster
  );

  // =========================================================================
  // Point-to-Point Operations
  // =========================================================================

  /**
   * Send data from one GPU to another
   * @param srcGpu - Source GPU index
   * @param srcBufferId - Source buffer ID
   * @param dstGpu - Destination GPU index
   * @param dstBufferId - Destination buffer ID (created if not provided)
   * @param sizeBytes - Bytes to transfer
   */
  send(
    srcGpu: number,
    srcBufferId: string,
    dstGpu: number,
    dstBufferId?: string,
    sizeBytes?: number
  ): Promise<P2PTransferResult>;

  /**
   * Copy data between GPUs (same as send, but creates destination buffer)
   */
  copy(
    srcGpu: number,
    srcBufferId: string,
    dstGpu: number,
    label?: string
  ): Promise<{ bufferRef: VirtualBufferRef; transfer: P2PTransferResult }>;

  // =========================================================================
  // Collective Operations
  // =========================================================================

  /**
   * All-reduce: Sum buffers across GPUs, result on all GPUs
   * @param bufferIds - Buffer IDs on each GPU (must all have same size)
   * @param gpuIndices - GPU indices participating
   */
  allReduce(
    bufferIds: Map<number, string>,
    gpuIndices: number[]
  ): Promise<CollectiveResult>;

  /**
   * All-gather: Gather buffers from all GPUs to all GPUs
   * Each GPU ends up with concatenation of all buffers
   * @param bufferIds - Buffer IDs on each GPU
   * @param gpuIndices - GPU indices participating
   */
  allGather(
    bufferIds: Map<number, string>,
    gpuIndices: number[]
  ): Promise<CollectiveResult>;

  /**
   * Reduce-scatter: Reduce then scatter result shards
   * @param bufferIds - Buffer IDs on each GPU
   * @param gpuIndices - GPU indices participating
   */
  reduceScatter(
    bufferIds: Map<number, string>,
    gpuIndices: number[]
  ): Promise<CollectiveResult>;

  /**
   * Broadcast: Send from one GPU to all others
   * @param srcGpu - Source GPU index
   * @param srcBufferId - Source buffer ID
   * @param gpuIndices - Destination GPU indices
   */
  broadcast(
    srcGpu: number,
    srcBufferId: string,
    gpuIndices: number[]
  ): Promise<CollectiveResult>;

  /**
   * Scatter: Distribute chunks from one GPU to all others
   * @param srcGpu - Source GPU with full buffer
   * @param srcBufferId - Source buffer ID
   * @param gpuIndices - Destination GPU indices
   */
  scatter(
    srcGpu: number,
    srcBufferId: string,
    gpuIndices: number[]
  ): Promise<CollectiveResult>;

  /**
   * Gather: Collect chunks from all GPUs to one GPU
   * @param bufferIds - Buffer IDs on each GPU
   * @param dstGpu - Destination GPU index
   */
  gather(
    bufferIds: Map<number, string>,
    dstGpu: number
  ): Promise<CollectiveResult>;

  // =========================================================================
  // Utility
  // =========================================================================

  /**
   * Get statistics for NVLink operations
   */
  getStats(): NVLinkFabricStats;

  /**
   * Reset statistics
   */
  resetStats(): void;

  /**
   * Check if two GPUs are in the same node
   * @param gpu1 - First GPU index
   * @param gpu2 - Second GPU index
   */
  sameNode(gpu1: number, gpu2: number): boolean;

  /**
   * Get effective bandwidth between two GPUs
   * Lower for cross-node transfers
   * @param gpu1 - First GPU index
   * @param gpu2 - Second GPU index
   */
  getEffectiveBandwidth(gpu1: number, gpu2: number): number;
}

/**
 * Statistics for NVLink fabric operations
 */
export interface NVLinkFabricStats {
  /** Total P2P transfers */
  totalP2PTransfers: number;
  /** Total collective operations */
  totalCollectives: number;
  /** Total bytes transferred */
  totalBytesTransferred: number;
  /** Total simulated time in ms */
  totalSimulatedTimeMs: number;
  /** Total actual time in ms */
  totalActualTimeMs: number;
  /** Breakdown by collective type */
  collectiveBreakdown: Map<string, { count: number; bytes: number; timeMs: number }>;
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create an NVLink fabric controller
 */
export declare function createNVLinkFabric(
  spec: NVLinkSpec,
  topology: EmulatedClusterTopology,
  timingModel: TimingModel,
  cluster: VirtualCluster
): NVLinkFabric;
