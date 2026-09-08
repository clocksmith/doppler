import { hashTargetPlan, assertQualifiedTargetOperation, normalizeTargetPlanSelectionPolicy } from '../../config/target-plan.js';
import { assertInitialExecutionIdentity } from '../../config/initial-execution-identity.js';
import { freezeCapsuleV2, verifyCapsuleV2Artifacts } from '../../config/capsule-v2.js';
import { verifyCapsuleMetadata, getCapsuleIdentity } from '../../config/capsule.js';
import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';
import { hashCapsuleSequenceInput, hashCapsuleSequenceOutput } from '../../config/capsule-sequence-receipt.js';
import { createVerifiedCapsuleArtifactStore } from './verified-capsule-artifact-store.js';
import { createResourceBinder } from './resource-binder.js';
import { createCommandExecutor } from './command-executor.js';
import { createSessionController } from './session-controller.js';
import { selectTargetPlan } from './target-selector.js';
import { executeCapsuleRerank } from './capsule-rerank.js';
import { createCapsuleOperationAdapters } from './capsule-operation-adapters.js';
import { createCapsuleAdapterExecution } from './capsule-adapter-execution.js';
import { createCapsuleOperationExecutor } from './capsule-operation-executor.js';
import { createCapsuleSessionExecution } from './capsule-session-execution.js';
import { executeCapsuleForecast } from './capsule-forecast.js';
import { executeCapsuleEmbedding } from './capsule-embedding.js';
import { CapsuleReleaseStateError } from '../../config/capsule-release-events.js';
import { createCapsuleReleaseAuthorization } from './capsule-release-authorization.js';
import { createCapsuleLoadScope, assertCapsuleLoadActive } from './capsule-acquisition.js';

export { createForecastProgramFactory } from './capsule-forecast-program.js';

export const RUNTIME_CORE_VERSION = '2.0.0';

function emit(observer, event) {
  observer?.observe?.(Object.freeze({ ...event }));
}

function withRequestSignal(request, signal) {
  // Leave malformed requests intact for the operation owner's validation.
  if (!request || typeof request !== 'object' || Array.isArray(request)) return request;
  const options = request.options;
  if (options !== undefined && (!options || typeof options !== 'object' || Array.isArray(options))) return request;
  if (options?.signal === null) return request;
  return { ...request, options: { ...options, signal } };
}

async function loadModuleSources(capsule, artifactStore) {
  if (typeof artifactStore?.readArtifact !== 'function') return new Map();
  const artifactById = new Map(capsule.artifacts.map((artifact) => [artifact.artifactId, artifact]));
  const modules = new Map();
  for (const module of capsule.wgslModules) {
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
  const { device, capsuleSource = null, artifactStore, cache = null, observer = null, trustedSigners, programFactory } = ports;

  return {
    version: RUNTIME_CORE_VERSION,
    ports: { device, capsuleSource, artifactStore, cache, observer },

    async openCapsule(capsuleOrId, options = {}) {
      const selectionPolicy = normalizeTargetPlanSelectionPolicy({
        acceptedTargetPlanDigests: options.acceptedTargetPlanDigests,
        requiredOperations: options.requiredOperations,
        preferredTargetPlanDigests: options.preferredTargetPlanDigests,
      });
      const acquisition = createCapsuleLoadScope(options);
      options = acquisition.options;
      let verifiedStore;
      let program;
      try {
        const input = typeof capsuleOrId === 'string'
          ? await capsuleSource?.fetchCapsule?.(capsuleOrId, options)
          : capsuleOrId;
        assertCapsuleLoadActive(options.signal);
        const capsule = freezeCapsuleV2(structuredClone(input));
        emit(observer, { type: 'capsule-validation-started', capsuleId: capsule.capsuleId });
        let verification;
        try {
          verification = await verifyCapsuleMetadata(capsule, { ...options, trustedSigners });
          if (verification.lifecycle) {
            if (typeof options.persistReleaseCheckpoint !== 'function') throw new Error('Capsule v3 requires persistReleaseCheckpoint before execution.');
            await options.persistReleaseCheckpoint(verification.lifecycle.checkpoint);
          }
        } catch (error) {
          if (error instanceof CapsuleReleaseStateError && typeof options.persistReleaseCheckpoint === 'function') {
            try { await options.persistReleaseCheckpoint(error.checkpoint); } catch (persistenceError) {
              throw new AggregateError([error, persistenceError], 'Release rejected; its verified checkpoint could not be persisted.', { cause: error });
            }
          }
          throw error;
        }
        assertCapsuleLoadActive(options.signal);
        const releaseAuthorization = createCapsuleReleaseAuthorization(verification.lifecycle);
        const deviceProfile = typeof device.getProfile === 'function'
          ? await device.getProfile()
          : {
              hasF16: Boolean(device.hasF16),
              hasSubgroups: Boolean(device.hasSubgroups),
              maxBufferSize: Number(device.maxBufferSize || 0),
            };
        assertCapsuleLoadActive(options.signal);
        const selectedPlan = selectTargetPlan(capsule.targetPlans, deviceProfile, selectionPolicy);
        const targetPlanDigest = hashTargetPlan(selectedPlan);
        emit(observer, { type: 'target-selected', capsuleId: capsule.capsuleId, targetId: selectedPlan.targetId, targetPlanDigest });
        verifiedStore = createVerifiedCapsuleArtifactStore(capsule, artifactStore, options);
        const artifactReceipts = await verifyCapsuleV2Artifacts(capsule, verifiedStore);
        verification = freezeCapsuleV2({ ...verification, artifactReceipts });
        await cache?.set?.(capsule.semanticRoot, {
          schema: 'doppler.capsule-verification-cache/v1', semanticRoot: capsule.semanticRoot, artifactReceipts,
        });
        assertCapsuleLoadActive(options.signal);
        emit(observer, { type: 'capsule-validation-complete', capsuleId: capsule.capsuleId, semanticRoot: capsule.semanticRoot,
          artifactMetrics: verifiedStore.getMetrics() });
        const modules = await loadModuleSources(capsule, verifiedStore);
        const manifestArtifact = capsule.artifacts.find((artifact) => artifact.artifactId === capsule.program.manifestArtifactId);
        const manifest = freezeCapsuleV2(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(await verifiedStore.readArtifact(manifestArtifact))));
        if (manifest.modelId !== capsule.modelId) throw new Error('Signed manifest model identity mismatch.');
        let observedInitialExecutionIdentity = null;
        program = await programFactory({ capsule, targetPlan: selectedPlan, artifactStore: verifiedStore, deviceProfile, options });
        assertCapsuleLoadActive(options.signal);
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
            capsuleId: capsule.capsuleId,
            targetId: selectedPlan.targetId,
            identityDigest: observedInitialExecutionIdentity.digest,
          });
        }
        const resourceBinder = createResourceBinder(device, program);
        const commandExecutor = createCommandExecutor(device, resourceBinder, program);
        const sessionController = createSessionController(commandExecutor, resourceBinder, program);
        const execution = createCapsuleSessionExecution();
        let closed = false;

        async function assertPlanUnchanged(observeProgram = true) {
          if (observeProgram) resourceBinder.assertDeviceAvailable();
          const observed = hashTargetPlan(selectedPlan);
          if (observed !== targetPlanDigest) {
            throw new Error(`Capsule Runtime mutated TargetPlan "${selectedPlan.targetId}" during execution.`);
          }
          if (getCapsuleIdentity(capsule).envelopeDigest !== verification.identity.envelopeDigest) throw new Error('Capsule Runtime mutated its executable closure.');
          if (observeProgram && selectedPlan.schema === 'doppler.target-plan/v2') {
            assertInitialExecutionIdentity(selectedPlan.initialExecutionIdentity, await program.getInitialExecutionIdentity());
          }
        }

        const session = {
          modelId: capsule.modelId,
          capsuleId: capsule.capsuleId,
          semanticRoot: capsule.semanticRoot,
          schema: 'doppler.capsule-session/v1',
          capsuleIdentity: verification.identity,
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
            if (closed) throw new Error('Capsule runtime session is closed.');
            releaseAuthorization.assertAssignment(request?.assignmentHash);
            await assertPlanUnchanged();
            assertQualifiedTargetOperation(selectedPlan, deviceProfile.surface, 'forecast');
            try {
              return releaseAuthorization.bindResult(await executeCapsuleForecast({ identity: verification.identity,
                release: verification.lifecycle?.release ?? capsule.release,
                targetPlan: selectedPlan, targetPlanDigest, program, request,
                artifactReceipts: verification.artifactReceipts,
                releaseEventDigest: verification.lifecycle?.event.digest ?? null }));
            } finally { await assertPlanUnchanged(); }
          },

          async embed(request) {
            if (closed) throw new Error('Capsule runtime session is closed.');
            await assertPlanUnchanged();
            assertQualifiedTargetOperation(selectedPlan, deviceProfile.surface, 'embed');
            try {
              return releaseAuthorization.bindResult(await executeCapsuleEmbedding({ identity: verification.identity,
                release: verification.lifecycle?.release ?? capsule.release,
                manifest, manifestHash: manifestArtifact.hash,
                targetPlan: selectedPlan, targetPlanDigest, program, request,
                artifactReceipts: verification.artifactReceipts,
                releaseEventDigest: verification.lifecycle?.event.digest ?? null }));
            } finally { await assertPlanUnchanged(); }
          },

          async encodeSequence(sequence, sequenceOptions = {}) {
            if (closed) throw new Error('Capsule runtime session is closed.');
            releaseAuthorization.assertAssignment(sequenceOptions.assignment);
            assertQualifiedTargetOperation(selectedPlan, deviceProfile.surface, 'encodeSequence');
            if (typeof program.encodeSequence !== 'function') throw new Error('Selected Capsule program does not implement sequence execution.');
            if (sequenceOptions.signal?.aborted) throw sequenceOptions.signal.reason ?? new Error('Sequence execution cancelled.');
            await assertPlanUnchanged();
            const { signal, ...requestOptions } = sequenceOptions;
            const executionOptions = { ...freezeCapsuleV2(structuredClone(requestOptions)), signal };
            const inputHash = hashCapsuleSequenceInput(sequence, executionOptions);
            const assignmentHash = executionOptions.assignment ? computeCanonicalSha256(executionOptions.assignment) : null;
            try {
              const result = await program.encodeSequence(sequence, executionOptions);
              if (sequenceOptions.signal?.aborted) throw sequenceOptions.signal.reason ?? new Error('Sequence execution cancelled.');
              const payload = {
                schema: 'doppler.capsule-execution-receipt/v1',
                operation: 'encodeSequence',
                capsule: verification.identity,
                targetId: selectedPlan.targetId,
                targetPlanDigest,
                artifactReceipts: verification.artifactReceipts,
                releaseEventDigest: verification.lifecycle?.event.digest ?? null,
                assignmentHash,
                inputHash,
                outputHash: hashCapsuleSequenceOutput(result),
              };
              return { ...result, receipt: releaseAuthorization.bindReceipt(freezeCapsuleV2({ ...payload, receiptDigest: computeCanonicalSha256(payload) })) };
            } finally { await assertPlanUnchanged(); }
          },

          resetGenerationState() {
            if (closed) throw new Error('Capsule runtime session is closed.');
            resourceBinder.assertDeviceAvailable();
            return program.reset?.();
          },

          async *generate(generationOptions = {}) {
            if (closed) throw new Error('Capsule runtime session is closed.');
            await assertPlanUnchanged();
            assertQualifiedTargetOperation(selectedPlan, deviceProfile.surface, 'generate');
            try {
              return yield* sessionController.generateTokens(selectedPlan, { ...generationOptions, modules });
            } finally {
              await assertPlanUnchanged();
            }
          },

          async generateText(generationOptions = {}) {
            const tokens = [];
            const iterator = this.generate(generationOptions);
            try {
              while (true) {
                const step = await iterator.next();
                if (step.done) return { text: program.decodeTokens(tokens), tokenIds: tokens, modelId: capsule.modelId, ...step.value };
                tokens.push(step.value);
              }
            } finally { await iterator.return?.(); }
          },

          async rerank(request) {
            if (closed) throw new Error('Capsule runtime session is closed.');
            await assertPlanUnchanged();
            assertQualifiedTargetOperation(selectedPlan, deviceProfile.surface, 'rerank');
            try {
              const receipt = releaseAuthorization.bindReceipt(await executeCapsuleRerank({
                capsule: verification.lifecycle ? { ...capsule, release: verification.lifecycle.release } : capsule,
                targetPlan: selectedPlan,
                targetPlanDigest,
                program,
                request,
              }));
              emit(observer, {
                type: 'capsule-rerank-complete',
                capsuleId: capsule.capsuleId,
                targetId: selectedPlan.targetId,
                receiptDigest: receipt.receiptDigest,
              });
              return receipt;
            } finally {
              await assertPlanUnchanged();
            }
          },

          close() {
            closed = true;
            return execution.close(async () => {
              try {
                await sessionController.close();
              } finally {
                commandExecutor.clearPipelineCache();
                verifiedStore.close();
                await assertPlanUnchanged(false);
              }
              emit(observer, { type: 'capsule-session-closed', capsuleId: capsule.capsuleId, targetPlanDigest });
            });
          },
        };
        // Internal adapters run under the caller's single execution lease.
        const local = { ...session };
        function requireBaseProgram() {
          if (program.getActiveAdapterIdentity?.()) throw new Error('Capsule adapter remains active; close and reopen the session.');
        }
        const runLocal = (task, signal) => execution.run(currentSignal => {
          requireBaseProgram();
          return task(currentSignal);
        }, signal);
        const adapters = createCapsuleOperationAdapters({ program,
          generate: (request) => local.generate(request), rerank: (request) => local.rerank(request),
          embed: (request) => local.embed(request),
          encodeSequence: (sequence, options) => local.encodeSequence(sequence, options) });
        assertCapsuleLoadActive(options.signal);
        emit(observer, { type: 'capsule-load-complete', capsuleId: capsule.capsuleId, artifactMetrics: verifiedStore.getMetrics() });
        const executeOperation = createCapsuleOperationExecutor({ adapters,
            prepareExecution: createCapsuleAdapterExecution({ program, capsule: { ...verification.identity, modelId: capsule.modelId }, targetPlan: selectedPlan }),
            identity: { capsule: verification.identity, modelId: capsule.modelId, targetId: selectedPlan.targetId, targetPlanDigest,
              artifactReceipts: verification.artifactReceipts, releaseEventDigest: verification.lifecycle?.event.digest ?? null,
              ...releaseAuthorization.receiptFields },
            async assertCurrent(request) {
              if (closed) throw new Error('Capsule runtime session is closed.');
              releaseAuthorization.assertAssignment(request.assignment);
              await assertPlanUnchanged();
            },
          });
        return Object.assign(session, {
          generate: (options = {}) => execution.stream(async function* (signal) {
            requireBaseProgram();
            return yield* local.generate({ ...options, signal });
          }, options.signal),
          generateText: async (options = {}) => runLocal(signal => local.generateText({ ...options, signal }), options.signal),
          forecast: async request => runLocal(signal => local.forecast({ ...request, signal }), request?.signal),
          embed: async request => runLocal(signal => local.embed(withRequestSignal(request, signal)), request?.options?.signal),
          rerank: async request => runLocal(signal => local.rerank(withRequestSignal(request, signal)), request?.options?.signal),
          encodeSequence: async (sequence, options = {}) => runLocal(signal => local.encodeSequence(sequence, { ...options, signal }), options.signal),
          resetGenerationState: () => runLocal(() => local.resetGenerationState()),
          executeOperation: (request, control = {}) => execution.stream(signal => executeOperation(request, { ...control, signal }), control.signal),
        });
      } catch (error) {
        acquisition.abort(error);
        try { await program?.close?.(); } catch (cleanupError) {
          throw new AggregateError([error, cleanupError], 'Capsule loading failed and program cleanup failed.', { cause: error });
        }
        throw error;
      } finally { acquisition.close(); if (acquisition.options.signal.aborted) verifiedStore?.close(); }
    },
  };
}
