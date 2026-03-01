import assert from 'node:assert/strict';
import { createTrainingConfig } from '../../src/config/training-defaults.js';

{
  const config = createTrainingConfig({
    training: {
      enabled: true,
      ul: {
        enabled: true,
        stage: 'stage1_joint',
      },
    },
  });
  assert.equal(config.training.ul.enabled, true);
  assert.equal(config.training.ul.stage, 'stage1_joint');
  assert.equal(config.training.ul.lambda0, 5);
}

{
  assert.throws(
    () => createTrainingConfig({
      training: {
        ul: {
          enabled: true,
          stage: 'unknown_stage',
        },
      },
    }),
    /UL config: stage must be one of stage1_joint, stage2_base/
  );
}

{
  assert.throws(
    () => createTrainingConfig({
      training: {
        ul: {
          enabled: true,
          stage: 'stage2_base',
          stage1Artifact: null,
        },
      },
    }),
    /UL config: stage2_base requires stage1Artifact/
  );
}

console.log('ul-training-schema.test: ok');
