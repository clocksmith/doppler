/**
 * Virtual Device Layer for NVIDIA Superchip Simulation
 *
 * Provides VirtualGPU and VirtualCluster abstractions that map
 * emulated GPU resources to local VRAM, RAM, and OPFS storage.
 *
 * @module simulator/virtual-device
 */

import type { EmulationConfigSchema, EmulatedGPUSpec, EmulatedCPUSpec } from '../config/schema/emulation.schema.js';

// =============================================================================
// Buffer Types
// =============================================================================

/**
 * Where a virtual buffer is currently stored
 */
export type BufferLocation =
  | 'vram'    // Actual WebGPU buffer (active working set)
  | 'ram'     // System RAM (ArrayBuffer staging)
  | 'opfs';   // OPFS storage (persistent, slow)

/**
 * Metadata for a virtual buffer
 */
export interface VirtualBufferMetadata {
  /** Unique buffer ID */
  id: string;
  /** Buffer size in bytes */
  sizeBytes: number;
  /** Current storage location */
  location: BufferLocation;
  /** Virtual GPU index that owns this buffer */
  gpuIndex: number;
  /** Label for debugging */
  label?: string;
  /** Timestamp of last access */
  lastAccessTime: number;
  /** Number of times this buffer has been accessed */
  accessCount: number;
  /** Whether buffer is currently pinned in VRAM */
  pinned: boolean;
}

/**
 * Reference to a virtual buffer
 */
export interface VirtualBufferRef {
  /** Buffer metadata */
  metadata: VirtualBufferMetadata;
  /** Get data from buffer (may trigger OPFS read) */
  getData(): Promise<ArrayBuffer>;
  /** Get actual WebGPU buffer (may trigger promotion to VRAM) */
  getGPUBuffer(): Promise<GPUBuffer>;
  /** Release buffer back to pool */
  release(): void;
}

// =============================================================================
// Virtual GPU
// =============================================================================

/**
 * Virtual GPU representing one emulated GPU's memory space
 */
export declare class VirtualGPU {
  /** GPU index in cluster */
  readonly index: number;
  /** GPU specification */
  readonly spec: EmulatedGPUSpec;
  /** OPFS path for this GPU's storage */
  readonly opfsPath: string;

  constructor(index: number, spec: EmulatedGPUSpec, opfsRootPath: string);

  /**
   * Allocate a virtual buffer on this GPU
   * @param sizeBytes - Size in bytes
   * @param label - Optional label for debugging
   */
  allocate(sizeBytes: number, label?: string): Promise<VirtualBufferRef>;

  /**
   * Write data to a buffer
   * @param bufferId - Buffer ID
   * @param data - Data to write
   * @param offset - Offset in bytes
   */
  write(bufferId: string, data: ArrayBuffer, offset?: number): Promise<void>;

  /**
   * Read data from a buffer
   * @param bufferId - Buffer ID
   * @param offset - Offset in bytes
   * @param length - Length in bytes (undefined = entire buffer)
   */
  read(bufferId: string, offset?: number, length?: number): Promise<ArrayBuffer>;

  /**
   * Free a virtual buffer
   * @param bufferId - Buffer ID
   */
  free(bufferId: string): Promise<void>;

  /**
   * Pin buffer in VRAM (prevent eviction)
   * @param bufferId - Buffer ID
   */
  pin(bufferId: string): Promise<void>;

  /**
   * Unpin buffer (allow eviction)
   * @param bufferId - Buffer ID
   */
  unpin(bufferId: string): Promise<void>;

  /**
   * Get buffer metadata
   * @param bufferId - Buffer ID
   */
  getBufferInfo(bufferId: string): VirtualBufferMetadata | null;

  /**
   * Get all buffer IDs
   */
  listBuffers(): string[];

  /**
   * Get memory statistics for this GPU
   */
  getMemoryStats(): VirtualGPUMemoryStats;

  /**
   * Evict least recently used buffers to free VRAM
   * @param bytesNeeded - Bytes to free
   */
  evictLRU(bytesNeeded: number): Promise<number>;

  /**
   * Initialize OPFS storage for this GPU
   */
  initialize(): Promise<void>;

  /**
   * Destroy and clean up all resources
   */
  destroy(): Promise<void>;
}

/**
 * Memory statistics for a virtual GPU
 */
export interface VirtualGPUMemoryStats {
  /** GPU index */
  gpuIndex: number;
  /** Total emulated VRAM */
  totalVramBytes: number;
  /** Bytes allocated (across all tiers) */
  allocatedBytes: number;
  /** Bytes currently in actual VRAM */
  vramUsedBytes: number;
  /** Bytes currently in RAM staging */
  ramUsedBytes: number;
  /** Bytes stored in OPFS */
  opfsUsedBytes: number;
  /** Number of active buffers */
  bufferCount: number;
  /** Number of pinned buffers */
  pinnedBufferCount: number;
}

// =============================================================================
// Virtual CPU (for Grace CPU memory)
// =============================================================================

/**
 * Virtual CPU representing Grace CPU memory space
 */
export declare class VirtualCPU {
  /** CPU index in cluster */
  readonly index: number;
  /** CPU specification */
  readonly spec: EmulatedCPUSpec;
  /** OPFS path for this CPU's storage */
  readonly opfsPath: string;

  constructor(index: number, spec: EmulatedCPUSpec, opfsRootPath: string);

  /**
   * Allocate a buffer in CPU memory space
   * @param sizeBytes - Size in bytes
   * @param label - Optional label
   */
  allocate(sizeBytes: number, label?: string): Promise<VirtualBufferRef>;

  /**
   * Write data to a buffer
   * @param bufferId - Buffer ID
   * @param data - Data to write
   */
  write(bufferId: string, data: ArrayBuffer): Promise<void>;

  /**
   * Read data from a buffer
   * @param bufferId - Buffer ID
   */
  read(bufferId: string): Promise<ArrayBuffer>;

  /**
   * Free a buffer
   * @param bufferId - Buffer ID
   */
  free(bufferId: string): Promise<void>;

  /**
   * Get memory statistics for this CPU
   */
  getMemoryStats(): VirtualCPUMemoryStats;

  /**
   * Initialize storage
   */
  initialize(): Promise<void>;

  /**
   * Destroy and clean up
   */
  destroy(): Promise<void>;
}

/**
 * Memory statistics for a virtual CPU
 */
export interface VirtualCPUMemoryStats {
  /** CPU index */
  cpuIndex: number;
  /** Total emulated memory */
  totalMemoryBytes: number;
  /** Bytes allocated */
  allocatedBytes: number;
  /** Number of buffers */
  bufferCount: number;
}

// =============================================================================
// Virtual Cluster
// =============================================================================

/**
 * Virtual cluster managing multiple GPUs and CPUs
 */
export declare class VirtualCluster {
  /** Cluster configuration */
  readonly config: EmulationConfigSchema;
  /** Virtual GPUs */
  readonly gpus: VirtualGPU[];
  /** Virtual CPUs */
  readonly cpus: VirtualCPU[];

  constructor(config: EmulationConfigSchema);

  /**
   * Initialize the cluster (creates OPFS directories, etc.)
   */
  initialize(): Promise<void>;

  /**
   * Get a specific virtual GPU
   * @param index - GPU index
   */
  getGPU(index: number): VirtualGPU;

  /**
   * Get a specific virtual CPU
   * @param index - CPU index
   */
  getCPU(index: number): VirtualCPU;

  /**
   * Get GPUs in a specific node
   * @param nodeIndex - Node index
   */
  getNodeGPUs(nodeIndex: number): VirtualGPU[];

  /**
   * Allocate a buffer on a specific GPU
   * @param gpuIndex - GPU index
   * @param sizeBytes - Size in bytes
   * @param label - Optional label
   */
  allocateOnGPU(gpuIndex: number, sizeBytes: number, label?: string): Promise<VirtualBufferRef>;

  /**
   * Allocate a sharded buffer across multiple GPUs (for tensor parallelism)
   * @param gpuIndices - GPU indices to shard across
   * @param totalSizeBytes - Total size (will be divided across GPUs)
   * @param label - Optional label
   */
  allocateSharded(
    gpuIndices: number[],
    totalSizeBytes: number,
    label?: string
  ): Promise<VirtualBufferRef[]>;

  /**
   * Get cluster-wide statistics
   */
  getClusterStats(): VirtualClusterStats;

  /**
   * Destroy and clean up all resources
   */
  destroy(): Promise<void>;
}

/**
 * Cluster-wide statistics
 */
export interface VirtualClusterStats {
  /** Number of GPUs */
  gpuCount: number;
  /** Number of CPUs */
  cpuCount: number;
  /** Total emulated VRAM across all GPUs */
  totalVramBytes: number;
  /** Total emulated CPU memory */
  totalCpuMemoryBytes: number;
  /** Total bytes allocated */
  totalAllocatedBytes: number;
  /** Bytes in actual VRAM */
  vramUsedBytes: number;
  /** Bytes in RAM staging */
  ramUsedBytes: number;
  /** Bytes in OPFS */
  opfsUsedBytes: number;
  /** Per-GPU stats */
  gpuStats: VirtualGPUMemoryStats[];
  /** Per-CPU stats */
  cpuStats: VirtualCPUMemoryStats[];
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a virtual cluster from emulation config
 * @param config - Emulation configuration
 */
export declare function createVirtualCluster(config: EmulationConfigSchema): VirtualCluster;

/**
 * Generate a unique buffer ID
 */
export declare function generateBufferId(): string;
