export declare function unifiedKernelWrapper(
  opName: string,
  target: GPUDevice | { device: GPUDevice; beginComputePass: unknown } | null,
  variant: string,
  bindings: unknown[],
  uniforms: Record<string, number>,
  workgroups: number | number[] | { indirectBuffer: GPUBuffer; indirectOffset?: number },
  constants?: Record<string, number | boolean> | null,
  extraBindings?: unknown[] | null,
  dispatchLabel?: string | null,
  signal?: AbortSignal | null
): Promise<boolean>;

export declare function withKernelOutput<T>(
  target: import('../command-recorder.js').CommandRecorder | GPUDevice | null,
  supplied: GPUBuffer | null, bytes: number, label: string,
  execute: (output: GPUBuffer) => Promise<T>
): Promise<T>;
