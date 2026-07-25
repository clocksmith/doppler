export const OBSERVATION_CONTEXT_SCHEMA = 'doppler.observation-context/v1';

function cloneJsonValue(value) {
  if (value == null) {
    return value;
  }
  return JSON.parse(JSON.stringify(value));
}

function deepFreeze(value) {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) {
    return value;
  }
  Object.freeze(value);
  for (const entry of Object.values(value)) {
    deepFreeze(entry);
  }
  return value;
}

export function createObservationContext(options = {}) {
  const runtimeConfig = options.runtimeConfig;
  if (!runtimeConfig?.shared?.tooling || !runtimeConfig?.shared?.debug) {
    throw new Error(
      '[ObservationContext] runtime.shared.tooling and runtime.shared.debug are required.'
    );
  }
  const debug = runtimeConfig.shared.debug;
  return deepFreeze({
    schema: OBSERVATION_CONTEXT_SCHEMA,
    commandContext: options.commandContext ?? null,
    diagnostics: runtimeConfig.shared.tooling.diagnostics,
    probes: cloneJsonValue(debug.probes ?? []),
    tracing: {
      pipelineEnabled: debug.pipeline?.enabled === true,
      layers: cloneJsonValue(debug.pipeline?.layers ?? null),
      kernelTrace: cloneJsonValue(debug.kernelTrace ?? null),
    },
    receiptPolicy: runtimeConfig.shared.tooling.refactorReceipts,
  });
}
