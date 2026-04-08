import assert from 'node:assert/strict';

import { assertImplicitDtypeTransitionAllowed } from '../../src/inference/pipelines/text/dtype-contract.js';

assert.doesNotThrow(() => {
  assertImplicitDtypeTransitionAllowed({
    executionPolicies: { dtypeTransition: 'require_cast_step' },
    fromDtype: 'f16',
    toDtype: 'f32',
    op: 'final_norm',
    transitionDeclaredBy: 'step_precision',
  });
}, 'explicit step precision should satisfy require_cast_step');

assert.throws(() => {
  assertImplicitDtypeTransitionAllowed({
    executionPolicies: { dtypeTransition: 'require_cast_step' },
    fromDtype: 'f16',
    toDtype: 'f32',
    op: 'final_norm',
  });
}, /explicit cast step/, 'undeclared transitions must still fail fast');

console.log('dtype-contract-step-precision.test: ok');
