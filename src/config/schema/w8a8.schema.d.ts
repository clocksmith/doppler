export interface W8A8Policy {
  schema: 'doppler.w8a8/v1';
  scheme: 'symmetric';
  weightGranularity: 'output-channel';
  activationGranularity: 'token';
  codeRange: readonly [-127, 127];
  inputDtype: 'f32';
  scaleDtype: 'f32';
  zeroScale: 1;
  rounding: 'nearest-even';
  inputPolicy: 'finite-normal-or-zero';
  scalePolicy: 'positive-normal';
  packing: 'w8a8-row-v1';
  accumulatorDtype: 'i32';
  maxReductionWidth: 133144;
  rescaleOrder: readonly ['activation', 'weight'];
  outputDtype: 'f32';
  quantizationRatio: 'exact-input-over-stored-scale';
  scaleRounding: 'nearest-even';
  rescaleRounding: 'nearest-even';
  outputSubnormals: 'flush-to-signed-zero-after-each-rescale';
}
export declare const W8A8_POLICY: Readonly<W8A8Policy>;
export declare function validateW8A8Policy(policy: unknown, label?: string): W8A8Policy;
