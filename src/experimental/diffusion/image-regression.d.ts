export interface DiffusionImageRegressionMetrics {
  samples: number;
  mae: number;
  mse: number;
  rmse: number;
  maxAbsDiff: number;
  psnr: number;
}

export interface DiffusionImageRegressionTolerance {
  maxAbsDiff?: number;
  mae?: number;
  rmse?: number;
  minPsnr?: number;
}

export declare function computeImageFingerprint(
  pixels: Uint8Array | Uint8ClampedArray
): string;

export declare function computeImageRegressionMetrics(
  actualPixels: Uint8Array | Uint8ClampedArray,
  expectedPixels: Uint8Array | Uint8ClampedArray
): DiffusionImageRegressionMetrics;

export declare function assertImageRegressionWithinTolerance(
  actualPixels: Uint8Array | Uint8ClampedArray,
  expectedPixels: Uint8Array | Uint8ClampedArray,
  tolerance?: DiffusionImageRegressionTolerance,
  contextLabel?: string
): DiffusionImageRegressionMetrics;
