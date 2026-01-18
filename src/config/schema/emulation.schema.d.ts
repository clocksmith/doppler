/**
 * Emulation Config Schema
 *
 * Configuration for NVIDIA superchip emulation using local resources.
 * Simulates GH200/GB200 superchips using actual VRAM, unified RAM, and OPFS storage.
 *
 * @module config/schema/emulation
 */

import type { EmulatedVramStats } from '../../storage/emulated-vram.js';

// =============================================================================
// Target Chip Types
// =============================================================================

/**
 * Supported emulated chip configurations
 */
export type EmulatedChipType =
  | 'gh200'        // Single GH200: 1 GPU (144GB) + 1 CPU (480GB)
  | 'gh200-nvl2'   // GH200 NVL2: 2 GPUs (288GB) + 2 CPUs (960GB)
  | 'gb200-8gpu'   // GB200 8-GPU Pod: 8 GPUs (1.5TB)
  | 'gb200-nvl72'; // GB200 NVL72: 72 GPUs (13.5TB) - future

/**
 * Timing simulation modes
 */
export type EmulationTimingMode =
  | 'functional'  // Verify distributed logic without timing delays
  | 'timed'       // Inject realistic delays for compute/memory/nvlink
  | 'hybrid';     // Timing for compute, functional for communication

// =============================================================================
// GPU Specifications
// =============================================================================

/**
 * Specification for a single GPU
 */
export interface EmulatedGPUSpec {
  /** GPU name (e.g., 'H100', 'H200', 'B200') */
  name: string;
  /** VRAM size in bytes */
  vramBytes: number;
  /** HBM bandwidth in bytes/sec */
  hbmBandwidthBytesPerSec: number;
  /** FP16 TFLOPS */
  fp16Tflops: number;
  /** FP8 TFLOPS (for B200) */
  fp8Tflops?: number;
}

/**
 * Specification for a CPU
 */
export interface EmulatedCPUSpec {
  /** CPU name (e.g., 'Grace') */
  name: string;
  /** CPU core count */
  cores: number;
  /** Memory size in bytes */
  memoryBytes: number;
  /** Memory bandwidth in bytes/sec */
  memoryBandwidthBytesPerSec: number;
}

// =============================================================================
// Interconnect Specifications
// =============================================================================

/**
 * NVLink interconnect specification
 */
export interface NVLinkSpec {
  /** Bandwidth in bytes/sec per link */
  bandwidthBytesPerSec: number;
  /** Base latency in microseconds */
  latencyUs: number;
}

/**
 * NVLink-C2C (CPU↔GPU coherent) specification
 */
export interface NVLinkC2CSpec {
  /** Bandwidth in bytes/sec */
  bandwidthBytesPerSec: number;
  /** Base latency in microseconds */
  latencyUs: number;
  /** Whether coherent memory is enabled */
  coherent: boolean;
}

// =============================================================================
// Cluster Topology
// =============================================================================

/**
 * Cluster topology configuration
 */
export interface EmulatedClusterTopology {
  /** Total GPU count in cluster */
  gpuCount: number;
  /** GPUs per node (for multi-node configs) */
  gpusPerNode: number;
  /** Node count (gpuCount / gpusPerNode) */
  nodeCount: number;
  /** CPU count (typically matches node count for superchips) */
  cpuCount: number;
}

// =============================================================================
// Parallelism Configuration
// =============================================================================

/**
 * Tensor parallelism configuration
 */
export interface TensorParallelConfig {
  /** Enable tensor parallelism */
  enabled: boolean;
  /** Degree of tensor parallelism (number of GPUs for weight sharding) */
  degree: number;
}

/**
 * Pipeline parallelism configuration
 */
export interface PipelineParallelConfig {
  /** Enable pipeline parallelism */
  enabled: boolean;
  /** Number of pipeline stages */
  stages: number;
  /** Micro-batch count for 1F1B schedule */
  microBatches?: number;
}

/**
 * Data parallelism configuration
 */
export interface DataParallelConfig {
  /** Enable data parallelism */
  enabled: boolean;
  /** Degree of data parallelism (number of model replicas) */
  degree: number;
}

/**
 * Expert parallelism configuration (for MoE models)
 */
export interface ExpertParallelConfig {
  /** Enable expert parallelism */
  enabled: boolean;
  /** Degree of expert parallelism */
  degree: number;
}

/**
 * Complete parallelism configuration
 */
export interface EmulatedParallelismConfig {
  tensorParallel: TensorParallelConfig;
  pipelineParallel: PipelineParallelConfig;
  dataParallel: DataParallelConfig;
  expertParallel: ExpertParallelConfig;
}

// =============================================================================
// Timing Scaling
// =============================================================================

/**
 * Timing scaling factors for emulation accuracy
 */
export interface EmulatedTimingScaling {
  /** Scale factor for compute operations (1.0 = real timing) */
  computeScale: number;
  /** Scale factor for memory operations */
  memoryScale: number;
  /** Scale factor for NVLink transfers */
  nvlinkScale: number;
}

// =============================================================================
// Resource Mapping
// =============================================================================

/**
 * Local resource tier configuration
 */
export interface LocalResourceTier {
  /** Tier name */
  name: string;
  /** Maximum bytes available in this tier */
  maxBytes: number;
  /** Approximate bandwidth in bytes/sec */
  bandwidthBytesPerSec: number;
}

/**
 * Local resource mapping for emulation
 */
export interface LocalResourceMapping {
  /** Tier 1: Actual GPU VRAM (WebGPU buffers) */
  tier1Vram: LocalResourceTier;
  /** Tier 2: System RAM (ArrayBuffer/WASM heap) */
  tier2Ram: LocalResourceTier;
  /** Tier 3: OPFS/SSD Storage */
  tier3Storage: LocalResourceTier;
}

// =============================================================================
// Main Emulation Config
// =============================================================================

/**
 * Complete emulation configuration schema
 */
export interface EmulationConfigSchema {
  /** Enable emulation mode */
  enabled: boolean;

  /** Target chip configuration to emulate */
  targetChip: EmulatedChipType;

  /** Timing simulation mode */
  timingMode: EmulationTimingMode;

  /** GPU specification (populated from preset or custom) */
  gpuSpec: EmulatedGPUSpec;

  /** CPU specification (for superchips with Grace CPU) */
  cpuSpec: EmulatedCPUSpec;

  /** Cluster topology */
  topology: EmulatedClusterTopology;

  /** NVLink GPU↔GPU specification */
  nvlink: NVLinkSpec;

  /** NVLink-C2C CPU↔GPU specification */
  nvlinkC2C: NVLinkC2CSpec;

  /** Parallelism configuration */
  parallelism: EmulatedParallelismConfig;

  /** Timing scaling factors */
  timingScaling: EmulatedTimingScaling;

  /** Local resource mapping (auto-detected or override) */
  localResources?: LocalResourceMapping;

  /** OPFS root path for emulated GPU memory partitions */
  opfsRootPath: string;

  /** Maximum active working set in bytes (fits in actual VRAM) */
  maxActiveWorkingSetBytes: number;

  /** Enable detailed emulation statistics */
  statsEnabled: boolean;

  /** Log emulation operations */
  logOperations: boolean;
}

// =============================================================================
// Emulation Statistics
// =============================================================================

/**
 * Statistics for a single virtual GPU
 */
export interface VirtualGPUStats {
  /** GPU index */
  gpuIndex: number;
  /** VRAM bytes allocated */
  vramAllocatedBytes: number;
  /** VRAM bytes in use */
  vramUsedBytes: number;
  /** Number of buffer allocations */
  allocationCount: number;
  /** Compute operations executed */
  computeOps: number;
  /** Total compute time (ms) */
  computeTimeMs: number;
}

/**
 * Statistics for NVLink transfers
 */
export interface NVLinkStats {
  /** Total bytes transferred */
  totalBytesTransferred: number;
  /** Number of transfers */
  transferCount: number;
  /** Total simulated transfer time (ms) */
  simulatedTimeMs: number;
  /** Actual transfer time (ms) */
  actualTimeMs: number;
}

/**
 * Complete emulation statistics
 */
export interface EmulationStats {
  /** Per-GPU statistics */
  gpuStats: VirtualGPUStats[];
  /** NVLink GPU↔GPU statistics */
  nvlinkStats: NVLinkStats;
  /** NVLink-C2C CPU↔GPU statistics */
  nvlinkC2CStats: NVLinkStats;
  /** Total injected delay (ms) */
  totalInjectedDelayMs: number;
  /** Wall clock time (ms) */
  wallClockTimeMs: number;
  /** Emulated VRAM store statistics (if available) */
  vramStore?: EmulatedVramStats;
}

// =============================================================================
// Default Exports
// =============================================================================

/** Default GH200 GPU spec (H200 variant) */
export declare const DEFAULT_GH200_GPU_SPEC: EmulatedGPUSpec;

/** Default GH200 CPU spec (Grace) */
export declare const DEFAULT_GH200_CPU_SPEC: EmulatedCPUSpec;

/** Default NVLink spec (900 GB/s) */
export declare const DEFAULT_NVLINK_SPEC: NVLinkSpec;

/** Default NVLink-C2C spec (900 GB/s coherent) */
export declare const DEFAULT_NVLINK_C2C_SPEC: NVLinkC2CSpec;

/** Default parallelism configuration (no parallelism) */
export declare const DEFAULT_PARALLELISM_CONFIG: EmulatedParallelismConfig;

/** Default emulation configuration (disabled) */
export declare const DEFAULT_EMULATION_CONFIG: EmulationConfigSchema;

/**
 * Create emulation config with overrides
 */
export declare function createEmulationConfig(
  overrides?: Partial<EmulationConfigSchema>
): EmulationConfigSchema;

/**
 * Get preset config for a specific chip type
 */
export declare function getChipPreset(chipType: EmulatedChipType): Partial<EmulationConfigSchema>;
