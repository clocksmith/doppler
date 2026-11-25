import assert from 'node:assert/strict';

import { assertImplicitDtypeTransitionAllowed } from "../../src/inference/pipelines/text/policy/dtype-contract.js";

assert.doesNotThrow(() => {
  assertImplicitDtypeTransitionAllowed({
    executionPolicies: { dtypeTransition: 'require-cast-step' },
    fromDtype: 'f16',
    toDtype: 'f32',
    op: 'final_norm',
    transitionDeclaredBy: 'step-precision',
  });
}, 'explicit step precision should satisfy require-cast-step');

assert.throws(() => {
  assertImplicitDtypeTransitionAllowed({
    executionPolicies: { dtypeTransition: 'require-cast-step' },
    fromDtype: 'f16',
    toDtype: 'f32',
    op: 'final_norm',
  });
}, /explicit cast step/, 'undeclared transitions must still fail fast');

console.log('dtype-contract-step-precision.test: ok');
