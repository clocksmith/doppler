import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, discardBuffer, uploadData } from '../../../../memory/buffer-pool.js';
import { isGpuBuffer } from './plan.js';
import { assertRecurrentStateLayout, assertConvolutionStateLayout } from './state-layout.js';

export const LINEAR_RUNTIME_SCHEMA_VERSION = 4;

/** @type {import('./state-clone.js').cloneLayerRuntimeState} */
export function cloneLayerRuntimeState(layerState) {
  assertRecurrentStateLayout(layerState);
  assertConvolutionStateLayout(layerState);
  return {
    layerIdx: layerState.layerIdx,
    seqLen: layerState.seqLen,
    warnedSeqMismatch: layerState.warnedSeqMismatch === true,
    convKernelSize: layerState.convKernelSize,
    convDim: layerState.convDim,
    keyDim: layerState.keyDim,
    valueDim: layerState.valueDim,
    numKHeads: layerState.numKHeads,
    numVHeads: layerState.numVHeads,
    headKDim: layerState.headKDim,
    headVDim: layerState.headVDim,
    qSize: layerState.qSize,
    kSize: layerState.kSize,
    vSize: layerState.vSize,
    qRep: layerState.qRep,
    normMode: layerState.normMode === 'per_head' ? 'per_head' : 'shared',
    rmsNormEps: layerState.rmsNormEps,
    convWeight: layerState.convWeight.slice(),
    dtBias: layerState.dtBias.slice(),
    aLog: layerState.aLog.slice(),
    normWeight: layerState.normWeight.slice(),
    convState: layerState.convState.slice(),
    convStatePrecision: layerState.convStatePrecision,
    convStateComponents: layerState.convStateComponents,
    normWeightOffset: layerState.normWeightOffset,
    recurrentState: layerState.recurrentState.slice(),
    recurrentStatePrecision: layerState.recurrentStatePrecision,
    recurrentStateComponents: layerState.recurrentStateComponents,
  };
}

/** @type {import('./state-clone.js').cloneLinearAttentionRuntimeOnGPU} */
export async function cloneLinearAttentionRuntimeOnGPU(runtime) {
  if (runtime == null) return { schemaVersion: LINEAR_RUNTIME_SCHEMA_VERSION, layers: new Map() };
  if (runtime.schemaVersion !== LINEAR_RUNTIME_SCHEMA_VERSION || !(runtime.layers instanceof Map)) {
    throw new Error('Linear runtime schema version mismatch; reset before reuse.');
  }
  const device = getDevice();
  if (!device) throw new Error('Resident recurrent clone requires a live GPU device.');
  /** @type {GPUBuffer[]} */
  const allocated = [];
  /** @type {import('../linear-attention.js').LinearAttentionRuntime} */
  const cloned = { schemaVersion: LINEAR_RUNTIME_SCHEMA_VERSION, layers: new Map() };
  try {
    const encoder = device.createCommandEncoder({ label: 'resident_linear_clone' });
    for (const [index, source] of runtime.layers) {
      const target = cloneLayerRuntimeState(source);
      cloned.layers.set(index, target);
      for (const field of /** @type {const} */ (['convWeight', 'dtBias', 'aLog', 'normWeight', 'convState', 'recurrentState'])) {
        const gpuField = /** @type {`${typeof field}GPU`} */ (`${field}GPU`);
        const bytes = target[field].byteLength;
        const sourceBuffer = source[gpuField];
        if (!bytes || (sourceBuffer && (!isGpuBuffer(sourceBuffer) || sourceBuffer.size < bytes))) {
          throw new Error(`Resident recurrent clone has invalid ${field} storage at layer ${index}.`);
        }
        const buffer = acquireBuffer(bytes, undefined, `L${index}.linear_clone_${field}`);
        allocated.push(buffer);
        target[gpuField] = buffer;
        if (sourceBuffer) encoder.copyBufferToBuffer(sourceBuffer, 0, buffer, 0, bytes);
        else uploadData(buffer, target[field]);
      }
    }
    if (allocated.length) {
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    return cloned;
  } catch (error) {
    try { await device.queue.onSubmittedWorkDone(); }
    catch (settlement) {
      for (const buffer of allocated) discardBuffer(buffer);
      throw new AggregateError([error, settlement], 'Resident recurrent clone settlement failed.');
    }
    for (const buffer of allocated) releaseBuffer(buffer);
    throw error;
  }
}
