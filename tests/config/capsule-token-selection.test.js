import assert from 'node:assert/strict';
import { validateCapsuleTokenSelection } from '../../src/config/capsule-token-selection.js';
import { createTargetPlanV2, hashTargetPlan } from '../../src/config/target-plan.js';
import { createSignedCapsuleFixture } from '../helpers/capsule-v2-fixture.js';
import { createInitialExecutionIdentity } from '../../src/config/initial-execution-identity.js';

const { capsule } = await createSignedCapsuleFixture();
const legacy = capsule.targetPlans[0];
const hash = legacy.executionGraphHash;
const base = createTargetPlanV2({ ...legacy, initialExecutionIdentity: createInitialExecutionIdentity({
  executionGraphHash: hash, resolvedGraphHash: hash, kernelClosure: [{ moduleId: 'main', file: 'main.wgsl', entry: 'main', digest: hash }],
  dtypeLane: { activation: 'f32', output: 'f32', kv: 'f32', math: 'f32', accumulation: 'f32' }, fusionSet: [],
  kvLayout: { layout: 'contiguous', kvDtype: 'f32' }, memoryPolicy: { kvcache: { layout: 'contiguous', kvDtype: 'f32' } },
  executionPlanDigest: hash, runtimeEngine: { schema: 'doppler.resolved-runtime-session/v1' },
}) });
const modules = ['sample', 'rep_penalty', 'logit_suppress'].map((id, i) => ({
  id, file: `${id}.wgsl`, digest: `sha256:${String(i + 1).repeat(64)}`,
  sourceHash: `sha256:${String(i + 4).repeat(64)}`,
}));
const declaration = { schema: 'doppler.capsule-token-selection/v1',
  generationContract: 'doppler.generation-contract/v1', logitsDtype: 'f32',
  kernelModules: modules.map(row => row.id) };
const plan = createTargetPlanV2({ ...base, tokenSelection: declaration,
  kernelClosure: [...base.kernelClosure, ...modules.map(({ id, digest, sourceHash }) => ({ moduleId: id, digest, sourceHash }))] });
assert.deepEqual(plan.tokenSelection, declaration);
assert.equal(validateCapsuleTokenSelection(plan, modules), declaration);
assert.notEqual(hashTargetPlan(plan), hashTargetPlan(base), 'adoption changes signed execution identity');
for (const changes of [{ schema: 'unknown' }, { generationContract: 'v2' }, { logitsDtype: 'f16' },
  { kernelModules: ['sample'] }, { kernelModules: ['sample', 'sample', 'sample'] },
  { kernelModules: ['sample', 'rep_penalty', 'unbound'] }, { fallback: 'cpu' }]) {
  assert.throws(() => createTargetPlanV2({ ...plan, tokenSelection: { ...declaration, ...changes } }), /token selection/);
}
assert.throws(() => validateCapsuleTokenSelection({ ...plan, schema: 'doppler.target-plan/v1' }), /v2/);
assert.throws(() => validateCapsuleTokenSelection(plan, modules.slice(1)), /outside/);
assert.throws(() => validateCapsuleTokenSelection(plan, modules.map(row => ({ ...row, file: 'other.wgsl' }))), /required sampling shader/);
assert.throws(() => validateCapsuleTokenSelection(plan, modules.map(row => ({ ...row, sourceHash: 'changed' }))), /identity mismatch/);
assert.equal(base.tokenSelection, undefined, 'legacy construction does not silently adopt GPU selection');
console.log('Capsule token selection contract tests passed.');
