import assert from 'node:assert/strict';

const { evaluateHotSwapRollout } = await import('../../src/hotswap/runtime.js');

{
  const decision = evaluateHotSwapRollout(
    {
      enabled: true,
      rollout: {
        mode: 'default',
        canaryPercent: 0,
        cohortSalt: 'test',
        optInAllowlist: [],
      },
    },
    { subjectId: 'user-1' }
  );
  assert.equal(decision.allowed, true);
  assert.equal(decision.reason, 'default_enabled');
}

{
  const policy = {
    enabled: true,
    rollout: {
      mode: 'canary',
      canaryPercent: 5,
      cohortSalt: 'stable-seed',
      optInAllowlist: [],
    },
  };
  const decisionA = evaluateHotSwapRollout(policy, { subjectId: 'stable-user' });
  const decisionB = evaluateHotSwapRollout(policy, { subjectId: 'stable-user' });
  assert.deepEqual(decisionA, decisionB);
}

{
  const decision = evaluateHotSwapRollout(
    {
      enabled: true,
      rollout: {
        mode: 'opt-in',
        canaryPercent: 0,
        cohortSalt: 'test',
        optInAllowlist: ['allow-user'],
      },
    },
    { subjectId: 'deny-user', optInTag: '' }
  );
  assert.strictEqual(decision.allowed, false);
  assert.equal(typeof decision.allowed, 'boolean');
  assert.equal(decision.reason, 'opt_in_required');
}

console.log('hotswap-rollout.test: ok');
