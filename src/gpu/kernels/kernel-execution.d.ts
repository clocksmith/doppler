export declare function unifiedKernelWrapper(
  opName: string,
  target: GPUDevice | { device: GPUDevice; beginComputePass: unknown } | null,
  variant: string,
  bindings: unknown[],
  uniforms: Record<string, number>,
  workgroups: number | [number, number, number],
  constants?: Record<string, number> | null,
  extraBindings?: unknown[] | null,
  dispatchLabel?: string | null
): Promise<void>;
