/**
 * Emulated VRAM Storage
 *
 * OPFS-backed storage for virtual GPU memory that simulates
 * HBM3e VRAM with tiered storage (actual VRAM → RAM → OPFS).
 *
 * @module storage/emulated-vram
 */

// =============================================================================
// Storage Tiers
// =============================================================================

/**
 * Storage tier for the emulated VRAM system
 */
export type StorageTier = 'vram' | 'ram' | 'opfs';

/**
 * Configuration for a storage partition
 */
export interface PartitionConfig {
  /** Partition name (e.g., 'gpu0', 'cpu0') */
  name: string;
  /** Maximum bytes for this partition */
  maxBytes: number;
  /** Parent directory path in OPFS */
  opfsPath: string;
}

// =============================================================================
// Chunk Management
// =============================================================================

/**
 * A chunk represents a fixed-size piece of virtual memory
 */
export interface MemoryChunk {
  /** Unique chunk ID */
  id: string;
  /** Chunk size in bytes */
  sizeBytes: number;
  /** Current storage tier */
  tier: StorageTier;
  /** Partition this chunk belongs to */
  partition: string;
  /** Whether chunk is locked (prevent eviction) */
  locked: boolean;
  /** Last access timestamp */
  lastAccessMs: number;
  /** Access count for statistics */
  accessCount: number;
}

/**
 * Result of a chunk read operation
 */
export interface ChunkReadResult {
  /** The data */
  data: ArrayBuffer;
  /** Tier the data was read from */
  fromTier: StorageTier;
  /** Time taken in ms */
  readTimeMs: number;
}

/**
 * Result of a chunk write operation
 */
export interface ChunkWriteResult {
  /** Tier the data was written to */
  toTier: StorageTier;
  /** Time taken in ms */
  writeTimeMs: number;
}

// =============================================================================
// Emulated VRAM Store
// =============================================================================

/**
 * OPFS-backed storage for emulated VRAM
 *
 * Manages tiered storage with automatic promotion/demotion
 * between actual VRAM, RAM, and OPFS.
 */
export declare class EmulatedVramStore {
  /**
   * Create a new emulated VRAM store
   * @param rootPath - Root path in OPFS for all partitions
   * @param vramBudgetBytes - Maximum bytes to keep in actual VRAM
   * @param ramBudgetBytes - Maximum bytes to keep in RAM
   */
  constructor(rootPath: string, vramBudgetBytes: number, ramBudgetBytes: number);

  /**
   * Initialize the store (creates OPFS directories)
   */
  initialize(): Promise<void>;

  /**
   * Create a new partition (e.g., for a virtual GPU)
   * @param config - Partition configuration
   */
  createPartition(config: PartitionConfig): Promise<void>;

  /**
   * Allocate a new chunk in a partition
   * @param partition - Partition name
   * @param sizeBytes - Chunk size
   * @param label - Optional label for debugging
   */
  allocate(partition: string, sizeBytes: number, label?: string): Promise<string>;

  /**
   * Write data to a chunk
   * @param chunkId - Chunk ID
   * @param data - Data to write
   * @param offset - Offset within chunk
   */
  write(chunkId: string, data: ArrayBuffer, offset?: number): Promise<ChunkWriteResult>;

  /**
   * Read data from a chunk
   * @param chunkId - Chunk ID
   * @param offset - Offset within chunk
   * @param length - Length to read (entire chunk if omitted)
   */
  read(chunkId: string, offset?: number, length?: number): Promise<ChunkReadResult>;

  /**
   * Free a chunk
   * @param chunkId - Chunk ID
   */
  free(chunkId: string): Promise<void>;

  /**
   * Lock a chunk to prevent eviction
   * @param chunkId - Chunk ID
   */
  lock(chunkId: string): Promise<void>;

  /**
   * Unlock a chunk to allow eviction
   * @param chunkId - Chunk ID
   */
  unlock(chunkId: string): Promise<void>;

  /**
   * Promote a chunk to a higher tier (OPFS → RAM → VRAM)
   * @param chunkId - Chunk ID
   * @param targetTier - Target tier
   */
  promote(chunkId: string, targetTier: StorageTier): Promise<void>;

  /**
   * Demote a chunk to a lower tier (VRAM → RAM → OPFS)
   * @param chunkId - Chunk ID
   * @param targetTier - Target tier
   */
  demote(chunkId: string, targetTier: StorageTier): Promise<void>;

  /**
   * Get chunk metadata
   * @param chunkId - Chunk ID
   */
  getChunkInfo(chunkId: string): MemoryChunk | null;

  /**
   * List all chunks in a partition
   * @param partition - Partition name
   */
  listChunks(partition: string): string[];

  /**
   * Get storage statistics
   */
  getStats(): EmulatedVramStats;

  /**
   * Evict chunks to free space in a tier
   * @param tier - Tier to free space in
   * @param bytesNeeded - Bytes to free
   */
  evict(tier: StorageTier, bytesNeeded: number): Promise<number>;

  /**
   * Destroy the store and clean up resources
   */
  destroy(): Promise<void>;
}

/**
 * Statistics for the emulated VRAM store
 */
export interface EmulatedVramStats {
  /** Total chunks across all partitions */
  totalChunks: number;
  /** Total bytes allocated */
  totalAllocatedBytes: number;
  /** Bytes in actual VRAM */
  vramUsedBytes: number;
  /** Bytes in RAM */
  ramUsedBytes: number;
  /** Bytes in OPFS */
  opfsUsedBytes: number;
  /** VRAM budget */
  vramBudgetBytes: number;
  /** RAM budget */
  ramBudgetBytes: number;
  /** Number of evictions performed */
  evictionCount: number;
  /** Total bytes evicted */
  totalBytesEvicted: number;
  /** Per-partition stats */
  partitionStats: PartitionStats[];
}

/**
 * Statistics for a single partition
 */
export interface PartitionStats {
  /** Partition name */
  name: string;
  /** Bytes allocated in this partition */
  allocatedBytes: number;
  /** Chunk count */
  chunkCount: number;
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create an emulated VRAM store with auto-detected budgets
 * @param rootPath - Root path in OPFS
 */
export declare function createEmulatedVramStore(rootPath: string): EmulatedVramStore;

/**
 * Detect available local resources for tier budgets
 */
export declare function detectLocalResources(): Promise<{
  vramBytes: number;
  ramBytes: number;
  storageBytes: number;
}>;
