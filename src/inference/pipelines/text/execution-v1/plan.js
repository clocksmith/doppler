function hasOwnProperty(value, key) {
  return value != null && Object.prototype.hasOwnProperty.call(value, key);
}

export function resolveRuntimeInferenceOverrideSection(runtimeOverrides, key) {
  const inferenceOverrides = runtimeOverrides?.inference;
  if (
    !inferenceOverrides
    || typeof inferenceOverrides !== 'object'
    || Array.isArray(inferenceOverrides)
  ) {
    return null;
  }
  if (!hasOwnProperty(inferenceOverrides, key)) {
    return null;
  }
  return inferenceOverrides[key] ?? null;
}

export function preserveRuntimeDecodeLoop(updatedInference, runtimeConfig) {
  const runtimeSession = runtimeConfig?.inference?.session;
  if (!hasOwnProperty(runtimeSession, 'decodeLoop')) {
    return updatedInference;
  }
  const updatedSession = updatedInference?.session;
  if (!updatedSession || typeof updatedSession !== 'object' || Array.isArray(updatedSession)) {
    return {
      ...updatedInference,
      session: {
        decodeLoop: runtimeSession.decodeLoop,
      },
    };
  }
  return {
    ...updatedInference,
    session: {
      ...updatedSession,
      decodeLoop: runtimeSession.decodeLoop,
    },
  };
}

export function preserveConfiguredKernelPath(updatedInference, runtimeConfig) {
  const configuredInference = runtimeConfig?.inference;
  const configuredKernelPath = configuredInference?.kernelPath;
  if (configuredKernelPath == null) {
    return updatedInference;
  }
  const configuredSession = configuredInference?.session;
  const hasConfiguredSessionCompute = hasOwnProperty(configuredSession, 'compute');
  const hasConfiguredSessionKVCache = hasOwnProperty(configuredSession, 'kvcache');
  return {
    ...updatedInference,
    kernelPath: configuredKernelPath,
    kernelPathSource: 'config',
    ...(hasOwnProperty(configuredInference, 'compute')
      ? { compute: configuredInference.compute }
      : {}),
    ...(hasConfiguredSessionCompute || hasConfiguredSessionKVCache
      ? {
          session: {
            ...updatedInference.session,
            ...(hasConfiguredSessionCompute ? { compute: configuredSession.compute } : {}),
            ...(hasConfiguredSessionKVCache ? { kvcache: configuredSession.kvcache } : {}),
          },
        }
      : {}),
  };
}
