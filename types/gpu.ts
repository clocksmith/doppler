/**
 * GPU and Kernel Types
 */

/** GPU device capabilities */
export interface GpuCapabilities {
  /** Supports shader-f16 extension */
  hasF16: boolean;
  /** Supports subgroups extension */
  hasSubgroups: boolean;
  /** Supports timestamp queries */
  hasTimestampQuery: boolean;
  /** Maximum buffer size */
  maxBufferSize: number;
  /** Maximum storage buffer binding size */
  maxStorageBufferBindingSize: number;
  /** Maximum compute workgroup size X */
  maxComputeWorkgroupSizeX: number;
  /** Maximum compute workgroup size Y */
  maxComputeWorkgroupSizeY: number;
  /** Maximum compute workgroup size Z */
  maxComputeWorkgroupSizeZ: number;
  /** Maximum compute invocations per workgroup */
  maxComputeInvocationsPerWorkgroup: number;
  /** Maximum compute workgroups per dimension */
  maxComputeWorkgroupsPerDimension: number;
  /** Subgroup size (if supported) */
  subgroupSize?: number;
  /** Minimum subgroup size */
  minSubgroupSize?: number;
  /** Maximum subgroup size */
  maxSubgroupSize?: number;
}

/** GPU device limits */
export interface GpuLimits {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  maxUniformBufferBindingSize: number;
  maxBindGroups: number;
  maxBindingsPerBindGroup: number;
  maxDynamicUniformBuffersPerPipelineLayout: number;
  maxDynamicStorageBuffersPerPipelineLayout: number;
}

/** Buffer handle with metadata */
export interface GpuBufferHandle {
  buffer: GPUBuffer;
  size: number;
  usage: GPUBufferUsageFlags;
  label?: string;
  pooled: boolean;
}

/** Buffer allocation request */
export interface BufferRequest {
  size: number;
  usage?: GPUBufferUsageFlags;
  label?: string;
  mappedAtCreation?: boolean;
}

/** Kernel execution mode */
export type KernelMode = 'immediate' | 'recorded' | 'batched';

/** Kernel executor interface */
export interface KernelExecutor {
  readonly mode: KernelMode;
  readonly device: GPUDevice;

  /** Execute a matmul operation */
  matmul(
    a: GPUBuffer,
    b: GPUBuffer,
    output: GPUBuffer,
    m: number,
    n: number,
    k: number,
    options?: MatmulOptions
  ): void | Promise<void>;

  /** Execute attention operation */
  attention(
    q: GPUBuffer,
    k: GPUBuffer,
    v: GPUBuffer,
    output: GPUBuffer,
    config: AttentionKernelConfig
  ): void | Promise<void>;

  /** Execute softmax operation */
  softmax(
    input: GPUBuffer,
    output: GPUBuffer,
    rows: number,
    cols: number,
    options?: SoftmaxOptions
  ): void | Promise<void>;

  /** Execute RMSNorm operation */
  rmsnorm(
    input: GPUBuffer,
    weight: GPUBuffer,
    output: GPUBuffer,
    rows: number,
    cols: number,
    eps: number
  ): void | Promise<void>;

  /** Execute RoPE operation */
  rope(
    input: GPUBuffer,
    output: GPUBuffer,
    config: RopeKernelConfig
  ): void | Promise<void>;

  /** Submit any pending commands */
  flush(): Promise<void>;
}

/** Matmul kernel options */
export interface MatmulOptions {
  transposeA?: boolean;
  transposeB?: boolean;
  alpha?: number;
  beta?: number;
}

/** Attention kernel configuration */
export interface AttentionKernelConfig {
  batchSize: number;
  seqLen: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  scale: number;
  causal: boolean;
  slidingWindow?: number;
}

/** Softmax kernel options */
export interface SoftmaxOptions {
  temperature?: number;
  inPlace?: boolean;
}

/** RoPE kernel configuration */
export interface RopeKernelConfig {
  seqLen: number;
  numHeads: number;
  headDim: number;
  ropeTheta: number;
  startPos: number;
}

/** Recorded command */
export interface RecordedCommand {
  type: string;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  workgroups: [number, number, number];
}

/** Command batch */
export interface CommandBatch {
  commands: RecordedCommand[];
  submit(): void;
}

/** Profiling event */
export interface ProfileEvent {
  name: string;
  startTime: number;
  endTime: number;
  duration: number;
  metadata?: Record<string, unknown>;
}

/** Profile session */
export interface ProfileSession {
  events: ProfileEvent[];
  totalDuration: number;
  kernelBreakdown: Map<string, number>;
}
