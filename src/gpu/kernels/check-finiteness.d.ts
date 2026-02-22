export declare const DEFAULT_FINITENESS_ABS_THRESHOLD: number;

export declare function resolveFinitenessAbsThreshold(value: number | null | undefined): number;

export declare function shouldTriggerFinitenessValue(value: number, absThreshold?: number): boolean;

export declare function recordCheckFiniteness(
  target: unknown,
  inputBuffer: GPUBuffer,
  size: number,
  statusBuffer: GPUBuffer,
  layerIdx?: number,
  step?: number,
  absThreshold?: number
): void;
