import { isTraceEnabled, trace } from '../../../../debug/index.js';
import { runProbes } from '../probes.js';

export function tracePrefillEmbeddingIds(embeddingInputIds, numTokens, embeddingOverride) {
  if (!isTraceEnabled('embed')) return;
  const overrideOffset = embeddingOverride?.offset ?? null;
  const overrideEnd = embeddingOverride
    ? embeddingOverride.offset + embeddingOverride.prefixLength - 1
    : null;
  trace.embed(
    `Prefill embedding IDs: tokens=${numTokens}, first=${embeddingInputIds[0] ?? 'missing'}, `
    + `last=${embeddingInputIds[numTokens - 1] ?? 'missing'}, overrideOffset=${overrideOffset ?? 'none'}, `
    + `overrideEnd=${overrideEnd ?? 'none'}, `
    + `offsetId=${overrideOffset == null ? 'none' : embeddingInputIds[overrideOffset]}, `
    + `afterOverrideId=${overrideEnd == null ? 'none' : embeddingInputIds[overrideEnd + 1] ?? 'missing'}`
  );
}

export async function probePrefillEmbedding(stage, tensor, options) {
  const { numTokens, hiddenSize, state, recorder } = options;
  await runProbes(stage, tensor.buffer, {
    numTokens,
    hiddenSize,
    probes: state.runtimeConfig.shared.debug.probes,
    recorder,
    operatorDiagnostics: state.operatorDiagnostics,
    dtype: tensor.dtype,
  });
}
