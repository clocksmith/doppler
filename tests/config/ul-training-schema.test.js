import assert from 'node:assert/strict';
import { createTrainingConfig } from '../../src/config/training-defaults.js';

{
  const config = createTrainingConfig({
    training: {
      enabled: true,
      ul: {
        enabled: true,
        stage: 'stage1-joint',
      },
    },
  });
  assert.equal(config.training.ul.enabled, true);
  assert.equal(config.training.ul.stage, 'stage1-joint');
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
    /UL config: stage must be one of stage1-joint, stage2-base/
  );
}

{
  assert.throws(
    () => createTrainingConfig({
      training: {
        ul: {
          enabled: true,
          stage: 'stage2-base',
          stage1Artifact: null,
        },
      },
    }),
    /UL config: stage2-base requires stage1Artifact/
  );
}

console.log('ul-training-schema.test: ok');
