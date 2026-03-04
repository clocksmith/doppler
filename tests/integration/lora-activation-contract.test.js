import assert from 'node:assert/strict';

const { activateLoRAFromTrainingOutput } = await import('../../src/client/doppler-provider/model-manager.js');

{
  const result = await activateLoRAFromTrainingOutput({
    adapterManifest: {
      id: 'demo',
      name: 'demo',
      baseModel: 'demo-base',
      rank: 4,
      alpha: 8,
      targetModules: ['q_proj'],
      tensors: [],
    },
  });
  assert.equal(result.activated, false);
  assert.equal(result.reason, 'no_model_loaded');
}

console.log('lora-activation-contract.test: ok');
