export type FP8Encoding = 'e4m3fn' | 'e5m2';
export interface FP8Policy {
  schema: 'doppler.fp8/v1';
  encoding: FP8Encoding;
  packing: 'fp8-block-v1';
  blockShape: [number, number];
  scaleMode: 'identity' | 'companion';
  scaleSuffix: string | null;
  scaleSemantics: 'multiply';
  scaleDtype: 'f32';
  inputDtype: 'f32';
  accumulatorDtype: 'f32';
  outputDtype: 'f32';
  dequantizedWeightPolicy: 'finite-normal-or-zero';
  nonfinite: 'preserve-storage-reject-gpu';
}
export function validateFP8Policy(input: unknown, label?: string): FP8Policy;
