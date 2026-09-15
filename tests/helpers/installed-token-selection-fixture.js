// Producer-side fixture construction. Synthetic execution; real shader bytes.
import fs from 'node:fs/promises';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PRIVATE_KEY, TEST_CAPSULE_PUBLIC_KEY } from './capsule-v2-fixture.js';
import { createInitialExecutionIdentityV2 } from '../../src/config/initial-execution-identity.js';
import { createTargetPlanV2 } from '../../src/config/target-plan.js';
import { buildCapsuleV2, signCapsuleV2 } from '../../src/config/capsule-v2.js';
import { sha256Hex } from '../../src/formats/sha256.js';

export async function createInstalledTokenSelectionFixture() {
  const base = await createSignedCapsuleFixture();
  const hash = text => `sha256:${sha256Hex(text)}`;
  const artifacts = [...base.capsule.artifacts], modules = [...base.capsule.wgslModules];
  const selectionIds = ['sample', 'rep_penalty', 'logit_suppress'];
  for (const id of selectionIds) {
    const file = `${id}.wgsl`, entry = id === 'sample' ? 'sample_single_pass' : 'main';
    const source = await fs.readFile(new URL(`../../src/gpu/kernels/${file}`, import.meta.url), 'utf8');
    const bytes = new TextEncoder().encode(source), sourceHash = hash(source);
    const digest = hash(`${source.replace(/\r\n/g, '\n')}\n@@entry:${entry}`);
    artifacts.push({ artifactId: id, role: 'wgsl-source', path: file, hash: sourceHash, sizeBytes: bytes.length });
    base.artifactBytes.set(id, bytes);
    modules.push({ id, file, entry, digest, sourceHash, sourceArtifactId: id });
  }
  const initialExecutionIdentity = createInitialExecutionIdentityV2({
    executionGraphHash: base.targetPlan.executionGraphHash, resolvedGraphHash: hash('synthetic graph'),
    kernelClosure: modules.map(({ id, file, entry, digest }) => ({ moduleId: id, file, entry, digest })),
    dtypeLane: { ...base.targetPlan.dtypes, output: 'f32' }, fusionSet: [], kvLayout: { layout: 'contiguous' },
    memoryPolicy: {}, executionPlanDigest: hash('synthetic plan'), runtimeEngine: { schema: 'injected-token-selection/v1' },
    programLoadPolicy: { schema: 'doppler.capsule-program-load-policy/v2', runtimeConfig: { inference: {
      session: {}, compute: {}, generation: { disableMultiTokenDecode: true },
    } } },
  });
  const targetPlan = createTargetPlanV2({ ...base.targetPlan, initialExecutionIdentity,
    kernelClosure: modules.map(({ id, digest, sourceHash }) => ({ moduleId: id, digest, sourceHash })),
    tokenSelection: { schema: 'doppler.capsule-token-selection/v1', generationContract: 'doppler.generation-contract/v1',
      logitsDtype: 'f32', kernelModules: selectionIds },
  });
  const capsule = await signCapsuleV2(buildCapsuleV2({ ...base.capsule, targetPlans: [targetPlan], artifacts, wgslModules: modules }), {
    authority: TEST_CAPSULE_AUTHORITY, privateKeyJwk: TEST_CAPSULE_PRIVATE_KEY, publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY,
  });
  return { capsule, trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
    artifacts: [...base.artifactBytes].map(([id, bytes]) => [id, [...bytes]]) };
}
