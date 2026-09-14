// Producer-side signing only. Consumers receive JSON and use public installed APIs.
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PRIVATE_KEY, TEST_CAPSULE_PUBLIC_KEY } from './capsule-v2-fixture.js';
import { createInitialExecutionIdentityV2 } from '../../src/config/initial-execution-identity.js';
import { createTargetPlanV2 } from '../../src/config/target-plan.js';
import { buildCapsuleV2, signCapsuleV2 } from '../../src/config/capsule-v2.js';
import { getCapsuleIdentity } from '../../src/config/capsule.js';

export async function createInstalledAdapterFixture() {
  const base = await createSignedCapsuleFixture();
  const hash = bytes => `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
  const artifacts = [...base.capsule.artifacts], modules = [...base.capsule.wgslModules];
  const kernelClosure = [];
  for (const [id, file] of [['matmul', 'matmul_f16.wgsl'], ['scale', 'scale.wgsl'], ['residual', 'residual.wgsl']]) {
    const bytes = new Uint8Array(await fs.readFile(new URL(`../../src/gpu/kernels/${file}`, import.meta.url)));
    const digest = hash(bytes);
    artifacts.push({ artifactId: id, role: 'wgsl-source', path: file, hash: digest, sizeBytes: bytes.length });
    base.artifactBytes.set(id, bytes);
    modules.push({ id, file, entry: 'main', digest, sourceHash: digest, sourceArtifactId: id });
    kernelClosure.push({ moduleId: id, file, entry: 'main', digest });
  }
  const initialExecutionIdentity = createInitialExecutionIdentityV2({
    executionGraphHash: base.targetPlan.executionGraphHash, resolvedGraphHash: hash('synthetic resolved graph'),
    kernelClosure, dtypeLane: base.targetPlan.dtypes, fusionSet: [], kvLayout: { layout: 'contiguous' },
    memoryPolicy: {}, executionPlanDigest: hash('synthetic plan'), runtimeEngine: { schema: 'injected-test-program/v1' },
    programLoadPolicy: { schema: 'doppler.capsule-program-load-policy/v2', runtimeConfig: { inference: {
      session: {}, compute: {}, generation: { disableMultiTokenDecode: true },
    } } },
  });
  const targetPlan = createTargetPlanV2({ ...base.targetPlan, initialExecutionIdentity,
    kernelClosure: kernelClosure.map(row => ({ moduleId: row.moduleId, digest: row.digest, sourceHash: row.digest })),
    adapterExecution: { schema: 'doppler.capsule-adapter-execution/v1', maxAdapters: 1, combination: 'single',
      formats: ['peft_safetensors'], operations: ['generate'], kernelModules: kernelClosure.map(row => row.moduleId) },
  });
  const capsule = await signCapsuleV2(buildCapsuleV2({ ...base.capsule, modelIR: base.modelIR,
    targetPlans: [targetPlan], artifacts, wgslModules: modules }), {
    authority: TEST_CAPSULE_AUTHORITY, privateKeyJwk: TEST_CAPSULE_PRIVATE_KEY, publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY,
  });
  const adapterBytes = [1, 2, 3, 4];
  const digest = hash(Uint8Array.from(adapterBytes));
  const adapter = { schema: 'doppler.capsule-adapter/v1', identity: hash('installed adapter'),
    baseModel: { ...getCapsuleIdentity(capsule), modelId: capsule.modelId },
    format: 'peft_safetensors', manifest: { id: 'installed-adapter', baseModel: capsule.modelId, rank: 1, alpha: 1,
      targetModules: ['q_proj'], checksum: digest, checksumAlgorithm: 'sha256', weightsFormat: 'safetensors',
      weightsPath: 'weights.safetensors', weightsSize: adapterBytes.length },
    artifact: { artifactId: 'installed-adapter', role: 'lora-weights', path: 'weights.safetensors', hash: digest, sizeBytes: adapterBytes.length },
  };
  return { capsule, trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
    artifacts: [...base.artifactBytes].map(([id, bytes]) => [id, [...bytes]]), adapter, adapterBytes };
}
