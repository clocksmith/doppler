import assert from 'node:assert/strict';

import {
  findNewModelFamilies,
  validateModelFamilyAuthorization,
} from '../../tools/check-model-family-intake.js';

assert.deepEqual(findNewModelFamilies(['gemma3', 'qwen3'], ['gemma3']), ['qwen3']);

const authorization = {
  schema: 'doppler.model-family-authorization/v1',
  family: 'customer-model',
  authority: 'customer',
  customerId: 'design-partner',
  applicationId: 'document-search',
  releaseContractPath: 'reports/customer/release.json',
  authorizationDigest: `sha256:${'a'.repeat(64)}`,
};
const release = {
  evidenceClass: 'external-candidate',
  claimBoundary: { externalCustomer: true },
  application: { applicationId: 'document-search' },
  rollout: { activationAuthority: 'customer' },
};
assert.deepEqual(
  validateModelFamilyAuthorization(authorization, 'customer-model', release, { ok: true }),
  ['authorization.authorizationDigest must match DOPPLER_MODEL_FAMILY_AUTHORIZATION']
);
assert.deepEqual(
  validateModelFamilyAuthorization(
    authorization,
    'customer-model',
    release,
    { ok: true },
    authorization.authorizationDigest
  ),
  []
);
assert.match(
  validateModelFamilyAuthorization(
    { ...authorization, authority: 'repository' },
    'customer-model',
    release,
    { ok: true },
    authorization.authorizationDigest
  )[0],
  /authority must be "customer"/
);

console.log('model-family-intake.test: ok');
