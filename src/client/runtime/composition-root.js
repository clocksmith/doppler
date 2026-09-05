import { hashTargetPlan } from '../../config/target-plan.js';
import { assertInitialExecutionIdentity } from '../../config/initial-execution-identity.js';
import { freezePackV2 } from '../../config/pack-v2.js';
import { validatePack, verifyPack, getPackIdentity } from '../../config/pack.js';
import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';
import { hashPackSequenceInput, hashPackSequenceOutput } from '../../config/pack-sequence-receipt.js';
import { createVerifiedPackArtifactStore } from './verified-pack-artifact-store.js';
import { createResourceBinder } from './resource-binder.js';
import { createCommandExecutor } from './command-executor.js';
import { createSessionController } from './session-controller.js';
import { selectTargetPlan } from './target-selector.js';
import { executePackRerank } from './pack-rerank.js';
import { executePackForecast } from './pack-forecast.js';

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
      const input = typeof packOrId === 'string'
        ? await packSource?.fetchPack?.(packOrId, options)
        : packOrId;
      const pack = freezePackV2(structuredClone(input));
      const structural = validatePack(pack);
      if (!structural.ok) throw new Error(`Invalid Doppler Pack: ${structural.errors.join('; ')}`);
      const verifiedStore = createVerifiedPackArtifactStore(pack, artifactStore);
      emit(observer, { type: 'pack-validation-started', packId: pack.packId });
      let verification;
      try {
        verification = await verifyPack(pack, { ...options, trustedSigners, artifactStore: verifiedStore });
        if (verification.lifecycle) {
          if (typeof options.persistReleaseCheckpoint !== 'function') throw new Error('Pack v3 requires persistReleaseCheckpoint before execution.');
          await options.persistReleaseCheckpoint(verification.lifecycle.checkpoint);
        }
      } catch (error) {
        verifiedStore.close();
        throw error;
      }
      let program;
      try {
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
        if (options.acceptedTargetPlanDigests !== undefined
          && (!Array.isArray(options.acceptedTargetPlanDigests)
            || !options.acceptedTargetPlanDigests.includes(targetPlanDigest))) {
          verifiedStore.close();
          throw new Error('Selected TargetPlan is not accepted by the application policy.');
        }
        emit(observer, { type: 'target-selected', packId: pack.packId, targetId: selectedPlan.targetId, targetPlanDigest });
        const modules = await loadModuleSources(pack, verifiedStore);
        const manifestArtifact = pack.artifacts.find((artifact) => artifact.artifactId === pack.program.manifestArtifactId);
        const manifest = freezePackV2(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(await verifiedStore.readArtifact(manifestArtifact))));
        if (manifest.modelId !== pack.modelId) throw new Error('Signed manifest model identity mismatch.');
        let observedInitialExecutionIdentity = null;
        program = await programFactory({ pack, targetPlan: selectedPlan, artifactStore: verifiedStore, deviceProfile, options });
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
        const resourceBinder = createResourceBinder(device, program);
        const commandExecutor = createCommandExecutor(device, resourceBinder, program);
        const sessionController = createSessionController(commandExecutor, resourceBinder, program);
        let closed = false;

        async function assertPlanUnchanged(observeProgram = true) {
          if (observeProgram) resourceBinder.assertDeviceAvailable();
          const observed = hashTargetPlan(selectedPlan);
          if (observed !== targetPlanDigest) {
            throw new Error(`Pack Runtime mutated TargetPlan "${selectedPlan.targetId}" during execution.`);
          }
          if (getPackIdentity(pack).envelopeDigest !== verification.identity.envelopeDigest) throw new Error('Pack Runtime mutated its executable closure.');
          if (observeProgram && selectedPlan.schema === 'doppler.target-plan/v2') {
            assertInitialExecutionIdentity(selectedPlan.initialExecutionIdentity, await program.getInitialExecutionIdentity());
          }
        }

        return {
          modelId: pack.modelId,
          packId: pack.packId,
          semanticRoot: pack.semanticRoot,
          schema: 'doppler.pack-session/v1',
          packIdentity: verification.identity,
          manifest,
          manifestHash: manifestArtifact.hash,
          get loaded() { return !closed; },
          get closed() { return closed; },
          selectedTargetId: selectedPlan.targetId,
          selectedTargetPlanDigest: targetPlanDigest,
          selectedPlan,
          deviceProfile,
          verification,
          observedInitialExecutionIdentity,
          units: { resourceBinder, commandExecutor, sessionController },

          async forecast(request) {
            if (closed) throw new Error('Pack runtime session is closed.');
            await assertPlanUnchanged();
            try {
              return await executePackForecast({ identity: verification.identity,
                release: verification.lifecycle?.release ?? pack.release,
                targetPlan: selectedPlan, targetPlanDigest, program, request,
                artifactReceipts: verification.artifactReceipts,
                releaseEventDigest: verification.lifecycle?.event.digest ?? null });
            } finally { await assertPlanUnchanged(); }
          },

          async encodeSequence(sequence, sequenceOptions = {}) {
            if (closed) throw new Error('Pack runtime session is closed.');
            if (typeof program.encodeSequence !== 'function') throw new Error('Selected Pack program does not implement sequence execution.');
            if (sequenceOptions.signal?.aborted) throw sequenceOptions.signal.reason ?? new Error('Sequence execution cancelled.');
            await assertPlanUnchanged();
            const { signal, ...requestOptions } = sequenceOptions;
            const executionOptions = { ...freezePackV2(structuredClone(requestOptions)), signal };
            const inputHash = hashPackSequenceInput(sequence, executionOptions);
            const assignmentHash = executionOptions.assignment ? computeCanonicalSha256(executionOptions.assignment) : null;
            try {
              const result = await program.encodeSequence(sequence, executionOptions);
              if (sequenceOptions.signal?.aborted) throw sequenceOptions.signal.reason ?? new Error('Sequence execution cancelled.');
              const payload = {
                schema: 'doppler.pack-execution-receipt/v1',
                operation: 'encodeSequence',
                pack: verification.identity,
                targetId: selectedPlan.targetId,
                targetPlanDigest,
                artifactReceipts: verification.artifactReceipts,
                releaseEventDigest: verification.lifecycle?.event.digest ?? null,
                assignmentHash,
                inputHash,
                outputHash: hashPackSequenceOutput(result),
              };
              return { ...result, receipt: freezePackV2({ ...payload, receiptDigest: computeCanonicalSha256(payload) }) };
            } finally { await assertPlanUnchanged(); }
          },

          resetGenerationState() {
            if (closed) throw new Error('Pack runtime session is closed.');
            resourceBinder.assertDeviceAvailable();
            return program.reset?.();
          },

          async *generate(generationOptions = {}) {
            if (closed) throw new Error('Pack runtime session is closed.');
            await assertPlanUnchanged();
            try {
              yield* sessionController.generateTokens(selectedPlan, { ...generationOptions, modules });
            } finally {
              await assertPlanUnchanged();
            }
          },

          async generateText(generationOptions = {}) {
            const tokens = [];
            for await (const tokenId of this.generate(generationOptions)) tokens.push(tokenId);
            return { text: program.decodeTokens(tokens), tokenIds: tokens };
          },

          async rerank(request) {
            if (closed) throw new Error('Pack runtime session is closed.');
            await assertPlanUnchanged();
            try {
              const receipt = await executePackRerank({
                pack: verification.lifecycle ? { ...pack, release: verification.lifecycle.release } : pack,
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
              await assertPlanUnchanged();
            }
          },

          async close() {
            if (closed) return;
            closed = true;
            try {
              await sessionController.close();
            } finally {
              commandExecutor.clearPipelineCache();
              verifiedStore.close();
              await assertPlanUnchanged(false);
            }
            emit(observer, { type: 'pack-session-closed', packId: pack.packId, targetPlanDigest });
          },
        };
      } catch (error) {
        try { await program?.close?.(); } finally { verifiedStore.close(); }
        throw error;
      }
    },
  };
}
