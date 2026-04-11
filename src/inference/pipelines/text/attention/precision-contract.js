import { getKernelPathAttentionPrecision } from '../../../../config/kernel-path-loader.js';

const VALID_DTYPES = new Set(['f16', 'f32']);

function normalizeOptionalAttentionDtype(value, label) {
  if (value == null || value === '') {
    return null;
  }
  const normalized = String(value).trim().toLowerCase();
  if (!VALID_DTYPES.has(normalized)) {
    throw new Error(`[ExecutionV1] ${label} must be "f16" or "f32"; got "${value}".`);
  }
  return normalized;
}

export function resolveAttentionPrecisionContract(config, state) {
  const phase = config?.isPrefill === true ? 'prefill' : 'decode';
  const layerIdx = Number.isInteger(config?.layerIdx) ? config.layerIdx : 0;
  const kernelPathPrecision = getKernelPathAttentionPrecision(phase, layerIdx, config?.kernelPath ?? undefined);
  const configActivationDtype = normalizeOptionalAttentionDtype(
    config?.activationDtype ?? null,
    `attention config activationDtype at layer ${layerIdx}`
  );
  const configInputDtype = normalizeOptionalAttentionDtype(
    config?.inputDtype ?? null,
    `attention config inputDtype at layer ${layerIdx}`
  );
  const configOutputDtype = normalizeOptionalAttentionDtype(
    config?.outputDtype ?? null,
    `attention config outputDtype at layer ${layerIdx}`
  );
  const kernelPathActivationDtype = normalizeOptionalAttentionDtype(
    kernelPathPrecision?.activationDtype ?? null,
    `attention precision activationDtype at layer ${layerIdx}`
  );
  const kernelPathInputDtype = normalizeOptionalAttentionDtype(
    kernelPathPrecision?.inputDtype ?? null,
    `attention precision inputDtype at layer ${layerIdx}`
  );
  const kernelPathOutputDtype = normalizeOptionalAttentionDtype(
    kernelPathPrecision?.outputDtype ?? null,
    `attention precision outputDtype at layer ${layerIdx}`
  );
  const configKvDtype = normalizeOptionalAttentionDtype(
    config?.kvDtype ?? null,
    `attention config kvDtype at layer ${layerIdx}`
  );
  const kernelPathKvDtype = normalizeOptionalAttentionDtype(
    kernelPathPrecision?.kvDtype ?? null,
    `attention precision kvDtype at layer ${layerIdx}`
  );

  const kernelPathResolvedInputDtype = kernelPathInputDtype ?? kernelPathActivationDtype ?? null;
  if (
    kernelPathActivationDtype
    && kernelPathInputDtype
    && kernelPathActivationDtype !== kernelPathInputDtype
  ) {
    throw new Error(
      `[ExecutionV1] attention precision declares activationDtype="${kernelPathActivationDtype}" ` +
      `but inputDtype="${kernelPathInputDtype}" at layer ${layerIdx}.`
    );
  }

  if (configInputDtype && kernelPathResolvedInputDtype && configInputDtype !== kernelPathResolvedInputDtype) {
    throw new Error(
      `[ExecutionV1] attention config declares inputDtype="${configInputDtype}" ` +
      `but kernelPath attention precision declares inputDtype="${kernelPathResolvedInputDtype}" at layer ${layerIdx}.`
    );
  }

  if (configOutputDtype && kernelPathOutputDtype && configOutputDtype !== kernelPathOutputDtype) {
    throw new Error(
      `[ExecutionV1] attention config declares outputDtype="${configOutputDtype}" ` +
      `but kernelPath attention precision declares outputDtype="${kernelPathOutputDtype}" at layer ${layerIdx}.`
    );
  }

  if (configKvDtype && kernelPathKvDtype && configKvDtype !== kernelPathKvDtype) {
    throw new Error(
      `[ExecutionV1] attention config declares kvDtype="${configKvDtype}" ` +
      `but kernelPath attention precision declares kvDtype="${kernelPathKvDtype}" at layer ${layerIdx}.`
    );
  }

  const explicitInputDtype = configInputDtype ?? kernelPathResolvedInputDtype ?? null;
  const explicitOutputDtype = configOutputDtype ?? kernelPathOutputDtype ?? null;
  const explicitKvDtype = configKvDtype ?? kernelPathKvDtype ?? null;
  const liveKvDtype = normalizeOptionalAttentionDtype(
    state?.kvCache?.kvDtype ?? null,
    `KV cache dtype at layer ${layerIdx}`
  );

  if (explicitKvDtype && liveKvDtype && explicitKvDtype !== liveKvDtype) {
    throw new Error(
      `[ExecutionV1] attention precision declares kvDtype="${explicitKvDtype}" ` +
      `but KV cache resolved "${liveKvDtype}" at layer ${layerIdx}.`
    );
  }

  return {
    precision: kernelPathPrecision,
    explicitInputDtype,
    explicitOutputDtype,
    explicitKvDtype,
    resolvedActivationDtype: explicitInputDtype ?? configActivationDtype ?? null,
    resolvedOutputDtype: explicitOutputDtype ?? configActivationDtype ?? null,
    resolvedKvCacheDtype: explicitKvDtype ?? liveKvDtype ?? null,
  };
}

export function isAttentionKvDtypeExplicit(contract, targetDtype) {
  const normalizedTarget = normalizeOptionalAttentionDtype(targetDtype, 'attention target kvDtype');
  return normalizedTarget != null && contract?.explicitKvDtype === normalizedTarget;
}
