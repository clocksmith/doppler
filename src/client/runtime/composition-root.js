import { hashTargetPlan } from '../../config/target-plan.js';
import { assertInitialExecutionIdentity } from '../../config/initial-execution-identity.js';
import { validatePackV2, verifyPackV2 } from '../../config/pack-v2.js';
import { createResourceBinder } from './resource-binder.js';
import { createCommandExecutor } from './command-executor.js';
import { createSessionController } from './session-controller.js';
import { selectTargetPlan } from './target-selector.js';
import { executePackRerank } from './pack-rerank.js';

export const RUNTIME_CORE_VERSION = '2.0.0';

function emit(observer, event) {
  observer?.observe?.(Object.freeze({ ...event }));
}

async function loadModuleSources(pack, artifactStore) {
  if (typeof artifactStore?.readArtifact !== 'function') return new Map();
  const artifactById = new Map(pack.artifacts.map((artifact) => [artifact.artifactId, artifact]));
  const modules = new Map();
  for (const module of pack.wgslModules) {
    const artifact = artifactById.get(module.sourceArtifactId);
    const bytes = await artifactStore.readArtifact(artifact);
    modules.set(module.id, { ...module, source: new TextDecoder().decode(bytes) });
  }
  return modules;
}

export function createDopplerRuntime(ports) {
  if (!ports || typeof ports !== 'object') throw new Error('createDopplerRuntime requires injected ports.');
  if (!ports.device) throw new Error('createDopplerRuntime requires a device port.');
  if (!ports.artifactStore) throw new Error('createDopplerRuntime requires an artifactStore port.');
  if (!ports.trustedSigners) throw new Error('createDopplerRuntime requires trustedSigners.');
  if (typeof ports.programFactory !== 'function') throw new Error('createDopplerRuntime requires programFactory.');
  const { device, packSource = null, artifactStore, cache = null, observer = null, trustedSigners, programFactory } = ports;

  return {
    version: RUNTIME_CORE_VERSION,
    ports: { device, packSource, artifactStore, cache, observer },

    async openPack(packOrId, options = {}) {
      const pack = typeof packOrId === 'string'
        ? await packSource?.fetchPack?.(packOrId, options)
        : packOrId;
      const structural = validatePackV2(pack);
      if (!structural.ok) throw new Error(`Invalid Doppler Pack v2: ${structural.errors.join('; ')}`);
      emit(observer, { type: 'pack-validation-started', packId: pack.packId });
      const verification = await verifyPackV2(pack, { trustedSigners, artifactStore });
      await cache?.set?.(pack.semanticRoot, {
        schema: 'doppler.pack-verification-cache/v1',
        semanticRoot: pack.semanticRoot,
        artifactReceipts: verification.artifactReceipts,
      });
      emit(observer, { type: 'pack-validation-complete', packId: pack.packId, semanticRoot: pack.semanticRoot });

      const deviceProfile = typeof device.getProfile === 'function'
        ? await device.getProfile()
        : {
            hasF16: Boolean(device.hasF16),
            hasSubgroups: Boolean(device.hasSubgroups),
            maxBufferSize: Number(device.maxBufferSize || 0),
          };
      const selectedPlan = selectTargetPlan(pack.targetPlans, deviceProfile);
      const targetPlanDigest = hashTargetPlan(selectedPlan);
      emit(observer, { type: 'target-selected', packId: pack.packId, targetId: selectedPlan.targetId, targetPlanDigest });
      const modules = await loadModuleSources(pack, artifactStore);
      const program = await programFactory({ pack, targetPlan: selectedPlan, artifactStore, deviceProfile, options });
      let observedInitialExecutionIdentity = null;
      try {
        if (selectedPlan.schema === 'doppler.target-plan/v2') {
          if (typeof program?.getInitialExecutionIdentity !== 'function') {
            throw new Error('TargetPlan v2 requires the loaded program to report initial execution identity.');
          }
          observedInitialExecutionIdentity = await program.getInitialExecutionIdentity();
          assertInitialExecutionIdentity(
            selectedPlan.initialExecutionIdentity,
            observedInitialExecutionIdentity
          );
          emit(observer, {
            type: 'initial-execution-identity-bound',
            packId: pack.packId,
            targetId: selectedPlan.targetId,
            identityDigest: observedInitialExecutionIdentity.digest,
          });
        }
      } catch (error) {
        await program?.close?.();
        throw error;
      }
      const resourceBinder = createResourceBinder(device, program);
      const commandExecutor = createCommandExecutor(device, resourceBinder, program);
      const sessionController = createSessionController(commandExecutor, resourceBinder, program);
      let closed = false;

      function assertPlanUnchanged() {
        const observed = hashTargetPlan(selectedPlan);
        if (observed !== targetPlanDigest) {
          throw new Error(`Pack Runtime mutated TargetPlan "${selectedPlan.targetId}" during execution.`);
        }
      }

      return {
        modelId: pack.modelId,
        packId: pack.packId,
        semanticRoot: pack.semanticRoot,
        selectedTargetId: selectedPlan.targetId,
        selectedTargetPlanDigest: targetPlanDigest,
        selectedPlan,
        deviceProfile,
        verification,
        observedInitialExecutionIdentity,
        units: { resourceBinder, commandExecutor, sessionController },

        async *generate(generationOptions = {}) {
          if (closed) throw new Error('Pack runtime session is closed.');
          try {
            yield* sessionController.generateTokens(selectedPlan, { ...generationOptions, modules });
          } finally {
            assertPlanUnchanged();
          }
        },

        async generateText(generationOptions = {}) {
          const tokens = [];
          for await (const tokenId of this.generate(generationOptions)) tokens.push(tokenId);
          return { text: program.decodeTokens(tokens), tokenIds: tokens };
        },

        async rerank(request) {
          if (closed) throw new Error('Pack runtime session is closed.');
          try {
            const receipt = await executePackRerank({
              pack,
              targetPlan: selectedPlan,
              targetPlanDigest,
              program,
              request,
            });
            emit(observer, {
              type: 'pack-rerank-complete',
              packId: pack.packId,
              targetId: selectedPlan.targetId,
              receiptDigest: receipt.receiptDigest,
            });
            return receipt;
          } finally {
            assertPlanUnchanged();
          }
        },

        async close() {
          if (closed) return;
          closed = true;
          await sessionController.close();
          commandExecutor.clearPipelineCache();
          assertPlanUnchanged();
          emit(observer, { type: 'pack-session-closed', packId: pack.packId, targetPlanDigest });
        },
      };
    },
  };
}
