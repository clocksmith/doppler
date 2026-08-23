export function queryScaleValidationError(value) {
  if (value === undefined) return null;
  return typeof value === 'number' && Number.isFinite(value) && value > 0
    ? null
    : 'attention.queryScale must be a positive finite number';
}

export function resolveQueryScale(value) {
  const resolved = value ?? 1;
  const error = queryScaleValidationError(resolved);
  if (error) throw new Error(error);
  return resolved;
}

export function ropeDisabledLayersValidationError(value) {
  if (value === undefined) return null;
  if (!Array.isArray(value)) return 'rope.disabledLayers must be an array of layer indices';
  return value.some((layerIdx) => !Number.isInteger(layerIdx) || layerIdx < 0)
    || new Set(value).size !== value.length
    ? 'rope.disabledLayers must contain unique non-negative integer layer indices'
    : null;
}

export function appendHeterogeneousAttentionValidation(errors, inf) {
  const queryScaleError = queryScaleValidationError(inf.attention.queryScale);
  if (queryScaleError) errors.push(queryScaleError);
  const disabledLayersError = ropeDisabledLayersValidationError(inf.rope.disabledLayers);
  if (disabledLayersError) errors.push(disabledLayersError);
}

export function resolveHeterogeneousAttentionContract(inf, numLayers, modelId) {
  const queryScale = resolveQueryScale(inf.attention.queryScale);
  const ropeDisabledLayers = inf.rope.disabledLayers ?? [];
  const disabledLayersError = ropeDisabledLayersValidationError(ropeDisabledLayers);
  if (disabledLayersError) throw new Error(disabledLayersError);
  if (ropeDisabledLayers.some((layerIdx) => layerIdx >= numLayers)) {
    throw new Error(
      `Manifest "${modelId}" declares rope.disabledLayers outside numLayers=${numLayers}.`
    );
  }
  return { queryScale, ropeDisabledLayers };
}

export function isRoPEDisabledForLayer(config, layerIdx) {
  return Array.isArray(config?.ropeDisabledLayers) && config.ropeDisabledLayers.includes(layerIdx);
}
