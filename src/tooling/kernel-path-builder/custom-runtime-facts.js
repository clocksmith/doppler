function normalizeLayerTypes(layerTypes) {
  if (!Array.isArray(layerTypes)) {
    return [];
  }
  return layerTypes.map((value) => String(value ?? '').trim().toLowerCase());
}

function findLayerIndexes(layerTypes, target) {
  const indexes = [];
  for (let index = 0; index < layerTypes.length; index += 1) {
    if (layerTypes[index] === target) {
      indexes.push(index);
    }
  }
  return indexes;
}

export function buildCustomRuntimeFacts(options = {}) {
  const modelId = String(options.modelId ?? 'model').trim() || 'model';
  const manifestInference = options.manifestInference && typeof options.manifestInference === 'object'
    ? options.manifestInference
    : {};
  const execution = manifestInference.execution && typeof manifestInference.execution === 'object'
    ? manifestInference.execution
    : {};
  const layerPattern = manifestInference.layerPattern && typeof manifestInference.layerPattern === 'object'
    ? manifestInference.layerPattern
    : {};
  const layerTypes = normalizeLayerTypes(layerPattern.layerTypes);
  const facts = [];

  const linearAttentionLayers = findLayerIndexes(layerTypes, 'linear_attention');
  if (linearAttentionLayers.length > 0) {
    facts.push({
      id: `${modelId}.linear_attention_runtime`,
      kind: 'custom-runtime',
      label: 'Linear attention runtime',
      summary: 'Linear-attention layers use recurrent runtime modules instead of raw kernel-path lowering, while remaining eligible for batched decode.',
      affectedLayers: linearAttentionLayers,
      assumptions: {
        registryBypass: true,
        recurrentState: true,
        projectionDtype: 'f16_or_f32',
        recurrentStateDtype: 'f32',
        decodeBatchingConstraint: 'batch-capable',
      },
      sourceRefs: [
        'src/inference/pipelines/text/linear-attention.js',
        'src/gpu/kernels/linear-attention-core.js',
      ],
    });
  }

  if (execution.inlineKernelPath === false) {
    facts.push({
      id: `${modelId}.execution_graph_only`,
      kind: 'execution-policy',
      label: 'Execution graph only',
      summary: 'The checked-in conversion contract owns an execution graph but explicitly opts out of inline kernel-path lowering.',
      assumptions: {
        inlineKernelPath: false,
        runtimeLowering: 'execution-graph-only',
      },
      sourceRefs: [
        'src/config/schema/execution-v1.schema.js',
        'src/inference/pipelines/text/execution-v1.js',
      ],
    });
  }

  return facts;
}
