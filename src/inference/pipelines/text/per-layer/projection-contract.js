import { createWeightBuffer, getBufferDtype, isGpuBufferInstance, isWeightBuffer } from "../../../../gpu/weight-buffer.js";

function isDensePleProjectionDtype(dtype) {
  return dtype === 'f16' || dtype === 'f32';
}

function resolvePleActivationStaticScaleValue(value, label) {
  if (value == null) {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(
      `Gemma 4 per-layer projection activation static scale "${label}" must be > 0. ` +
      `Got "${String(value)}".`
    );
  }
  return parsed;
}

export function resolvePleActivationStaticQdqConfig(perLayerInputWeights) {
  const inputScale = resolvePleActivationStaticScaleValue(
    perLayerInputWeights?.perLayerModelProjectionInputActivationStaticScale,
    'input'
  );
  const outputScale = resolvePleActivationStaticScaleValue(
    perLayerInputWeights?.perLayerModelProjectionOutputActivationStaticScale,
    'output'
  );
  if (inputScale == null && outputScale == null) {
    return null;
  }
  if (inputScale == null || outputScale == null) {
    throw new Error(
      'Gemma 4 per-layer projection static activation quantization requires both input and output scales.'
    );
  }
  return {
    inputScale,
    outputScale,
  };
}

export function resolveDensePleProjectionWeight(
  weight,
  label = 'per_layer_model_projection'
) {
  if (isWeightBuffer(weight)) {
    const dtype = String(weight.dtype ?? '').toLowerCase();
    if (!isDensePleProjectionDtype(dtype)) {
      throw new Error(
        'Gemma 4 sliced per-layer projection requires a dense f16/f32 base materialization. ' +
        `Got dtype="${String(weight.dtype)}" for "${label}".`
      );
    }
    if (!weight.materializations?.q4k?.buffer) {
      return weight;
    }
    return createWeightBuffer(
      weight.buffer,
      dtype,
      weight.layout,
      [...weight.shape],
      weight.label ?? label
    );
  }

  if (isGpuBufferInstance(weight)) {
    const dtype = String(getBufferDtype(weight) ?? '').toLowerCase();
    if (!isDensePleProjectionDtype(dtype)) {
      throw new Error(
        'Gemma 4 sliced per-layer projection requires dense f16/f32 GPU weights with runtime dtype metadata. ' +
        `Got dtype="${String(getBufferDtype(weight) ?? 'unknown')}" for "${label}".`
      );
    }
  }

  return weight;
}
