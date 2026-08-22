import { createResourceBinder } from './resource-binder.js';
import { createCommandExecutor } from './command-executor.js';
import { createSessionController } from './session-controller.js';
import { selectTargetPlan } from './target-selector.js';

export const RUNTIME_CORE_VERSION = '1.0.0';

/**
 * Creates an uncreative, deterministic Doppler Runtime instance bound to injected ports.
 *
 * @param {object} ports
 * @param {object} ports.device WebGPU device and capabilities provider
 * @param {object} [ports.packSource] Pack resolver/fetcher
 * @param {object} [ports.artifactStore] Shard and weights loader
 * @param {object} [ports.cache] Persistent cache adapter
 * @param {object} [ports.observer] Metrics and telemetry sink
 * @returns {object} Doppler runtime instance
 */
export function createDopplerRuntime(ports) {
  if (!ports || typeof ports !== 'object') {
    throw new Error('createDopplerRuntime requires an injected ports object.');
  }
  if (!ports.device) {
    throw new Error('createDopplerRuntime requires an injected device port.');
  }

  const { device, packSource = null, artifactStore = null, cache = null, observer = null } = ports;
  const resourceBinder = createResourceBinder(device);
  const commandExecutor = createCommandExecutor(device, resourceBinder);
  const sessionController = createSessionController(commandExecutor, resourceBinder);

  return {
    version: RUNTIME_CORE_VERSION,
    ports: {
      device,
      packSource,
      artifactStore,
      cache,
      observer,
    },
    units: {
      resourceBinder,
      commandExecutor,
      sessionController,
    },

    /**
     * Resolves and prepares a model session from a Doppler Pack.
     *
     * @param {string|object} packOrId
     * @param {object} [options]
     * @returns {Promise<object>}
     */
    async openPack(packOrId, options = {}) {
      let pack = packOrId;
      if (typeof packOrId === 'string' && packSource?.fetchPack) {
        pack = await packSource.fetchPack(packOrId, options);
      }
      if (!pack || typeof pack !== 'object') {
        throw new Error(`Failed to resolve valid Doppler Pack for: ${packOrId}`);
      }

      // 1. Inspect device capabilities
      const deviceProfile = typeof device.getProfile === 'function'
        ? await device.getProfile()
        : { hasF16: Boolean(device.hasF16), hasSubgroups: Boolean(device.hasSubgroups) };

      // 2. Select pre-qualified TargetPlan (no mutation)
      const targetPlans = Array.isArray(pack.targetPlans)
        ? pack.targetPlans
        : (pack.execution ? [{
            targetId: 'legacy-execution-v1',
            modelId: pack.modelId,
            capabilityPredicate: { requiresF16: false, requiresSubgroups: false, minBufferSize: 0 },
            dtypes: { activation: 'f32', kv: 'f32', weight: 'f32' },
            kernelClosure: pack.wgslModules || [],
            memoryLayout: { kvCacheLayout: 'contiguous', estimatedPeakBytes: 0 },
            phases: pack.execution,
          }] : []);

      const selectedPlan = selectTargetPlan(targetPlans, deviceProfile);

      return {
        modelId: pack.modelId,
        bundleId: pack.bundleId || pack.packId,
        selectedTargetId: selectedPlan.targetId,
        selectedPlan,
        deviceProfile,
        generate(generationOptions = {}) {
          return sessionController.generateTokens(selectedPlan, generationOptions);
        },
      };
    },
  };
}
