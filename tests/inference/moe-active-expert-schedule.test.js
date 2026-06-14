import assert from 'node:assert/strict';

import { DEFAULT_MOE_RUNTIME_CONFIG } from '../../src/config/schema/moe.schema.js';
import { buildActiveExpertScheduleFromIndices } from '../../src/inference/pipelines/text/moe-gpu.js';

{
  assert.equal(DEFAULT_MOE_RUNTIME_CONFIG.routing.activeExpertSelection, 'all');
  assert.doesNotThrow(() => buildActiveExpertScheduleFromIndices(
    new Uint32Array([0]),
    1,
    1,
    'topk-readback'
  ));
}

{
  const schedule = buildActiveExpertScheduleFromIndices(
    new Uint32Array([2, 0, 2, 3, 0]),
    4,
    3
  );

  assert.equal(schedule.selection, 'topk-readback');
  assert.deepEqual(schedule.activeExperts, [0, 2, 3]);
  assert.deepEqual(Array.from(schedule.tokenCounts), [2, 0, 2, 1]);
}

{
  assert.throws(
    () => buildActiveExpertScheduleFromIndices(new Uint32Array([1, 1, 1]), 2, 2),
    /Expert 1 received 3 tokens but maxTokensPerExpert=2/
  );
}

{
  assert.throws(
    () => buildActiveExpertScheduleFromIndices(new Uint32Array([0, 2]), 2, 2),
    /outside numExperts=2/
  );
}

console.log('moe-active-expert-schedule.test: ok');
