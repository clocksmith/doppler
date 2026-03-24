import assert from 'node:assert/strict';

const { buildCustomRuntimeFacts } = await import('../../src/tooling/kernel-path-builder/custom-runtime-facts.js');

{
  const facts = buildCustomRuntimeFacts({
    modelId: 'qwen-test',
    manifestInference: {
      layerPattern: {
        layerTypes: ['linear_attention', 'full_attention', 'linear_attention'],
      },
      execution: {
        inlineKernelPath: false,
      },
    },
  });

  const linearRuntimeFact = facts.find((fact) => fact.id === 'qwen-test.linear_attention_runtime');
  assert.ok(linearRuntimeFact, 'linear-attention runtime fact should be emitted');
  assert.equal(
    linearRuntimeFact.summary,
    'Linear-attention layers use recurrent runtime modules instead of raw kernel-path lowering, while remaining eligible for batched decode.'
  );
  assert.deepEqual(linearRuntimeFact.affectedLayers, [0, 2]);
  assert.deepEqual(linearRuntimeFact.assumptions, {
    registryBypass: true,
    recurrentState: true,
    projectionDtype: 'f16_or_f32',
    recurrentStateDtype: 'f32',
    decodeBatchingConstraint: 'batch-capable',
  });

  const executionFact = facts.find((fact) => fact.id === 'qwen-test.execution_graph_only');
  assert.ok(executionFact, 'execution-graph-only fact should be emitted when inline lowering is disabled');
  assert.equal(executionFact.assumptions.inlineKernelPath, false);
}

{
  const facts = buildCustomRuntimeFacts({
    modelId: 'full-only',
    manifestInference: {
      layerPattern: {
        layerTypes: ['full_attention'],
      },
      execution: {
        inlineKernelPath: true,
      },
    },
  });

  assert.equal(facts.find((fact) => fact.id === 'full-only.linear_attention_runtime'), undefined);
  assert.equal(facts.find((fact) => fact.id === 'full-only.execution_graph_only'), undefined);
}

console.log('custom-runtime-facts.test: ok');
