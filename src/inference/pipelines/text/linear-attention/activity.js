/** @type {typeof import('./activity.js').observeLinearActivity} */
export async function observeLinearActivity(activity, layerIdx, state, layout, a, b, numTokens, bOffset, recorder) {
  if (!activity) return;
  /** @type {Array<[string, GPUBuffer | null | undefined, number[], string | undefined]>} */
  const states = [
    ['recurrent-memory', state.recurrentStateGPU, [layout.numVHeads, state.headKDim, state.headVDim], state.recurrentStatePrecision],
    ['convolution-history', state.convStateGPU, [state.convKernelSize, state.convDim], state.convStatePrecision],
  ];
  for (const [role, buffer, shape, precision] of states) {
    if (!buffer) { activity.unavailable(role, 'GPU state buffer unavailable.'); continue; }
    await activity.capture(role, buffer, { layer: layerIdx, dtype: 'f32', shape, recorder, state: true,
      metadata: { boundary: 'after-chunk', positionsProcessed: numTokens },
      ...(precision === 'two-f32' ? { precision, components: 2, elementStrideBytes: 8 } : {}) });
  }
  /** @type {Array<[string, import('../../../../gpu/tensor.js').Tensor, number]>} */
  const projections = [['linear-a', a, 0], ['linear-b', b, bOffset]];
  for (const [role, tensor, offset] of projections) {
    await activity.capture(role, tensor.buffer, { layer: layerIdx, dtype: tensor.dtype, recorder,
      shape: [numTokens, layout.numVHeads], offset,
      precision: tensor.precision, components: tensor.components, elementStrideBytes: tensor.elementStrideBytes });
  }
  /** @type {Array<[string, GPUBuffer | null | undefined]>} */
  const parameters = [['linear-a-log', state.aLogGPU], ['linear-dt-bias', state.dtBiasGPU]];
  for (const [role, buffer] of parameters) {
    if (!buffer) continue;
    await activity.capture(role, buffer, { layer: layerIdx, dtype: 'f32', recorder,
      shape: [1, layout.numVHeads], metadata: { static: true } });
  }
}
