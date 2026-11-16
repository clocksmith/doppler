import { log } from "../../../../debug/index.js";

function resolveErrorMessage(error) {
  if (error && typeof error === 'object' && typeof error.message === 'string') {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return String(error);
}

// ============================================================================
// Emulation Setup
// ============================================================================

export async function initEmulation(runtimeConfig) {
  const emulationConfig = runtimeConfig?.emulation;

  // Skip if emulation is not enabled
  if (!emulationConfig?.enabled) {
    return null;
  }

  try {
    const simulatorModuleRoot = '/proto/simulator';
    const simulatorEnvSpecifier = `${simulatorModuleRoot}/env.js`;
    const simulatorIndexSpecifier = `${simulatorModuleRoot}/index.js`;

    // Dynamically import to avoid loading emulation code when disabled
    const { setSimulatorEnv } = await import(simulatorEnvSpecifier);
    const { createEmulationConfig, formatBytes, formatBandwidth } = await import("../../../../config/schema/platform/emulation.schema.js");
    const { EmulatedVramStore, detectLocalResources } = await import("../../../../storage/emulated-vram.js");
    const { getBufferPool } = await import("../../../../memory/buffer-pool.js");
    const { createEmulationContext, isEmulationSupported } = await import(simulatorIndexSpecifier);

    setSimulatorEnv({
      log,
      bufferPool: getBufferPool,
      createEmulationConfig,
      formatBytes,
      formatBandwidth,
      detectLocalResources,
      createVramStore: (config, budgets) =>
        new EmulatedVramStore(config.opfsRootPath, budgets.vramBudgetBytes, budgets.ramBudgetBytes),
    });

    const supported = await isEmulationSupported();
    if (!supported) {
      throw new Error('Emulation requested but not supported in this environment.');
    }

    // Create emulation context
    log.info('Pipeline', `Initializing emulation for ${emulationConfig.targetChip}`);
    const ctx = await createEmulationContext(emulationConfig);

    log.info('Pipeline', `Emulation ready: ${ctx.config.topology.gpuCount} virtual GPUs, timing mode: ${ctx.config.timingMode}`);

    return ctx;
  } catch (err) {
    const message = resolveErrorMessage(err);
    log.error('Pipeline', `Failed to initialize emulation: ${message}`);
    throw new Error(`Failed to initialize emulation: ${message}`);
  }
}

export async function destroyEmulation(emulation) {
  if (emulation) {
    try {
      await emulation.destroy();
      log.info('Pipeline', 'Emulation context destroyed');
    } catch (err) {
      log.warn('Pipeline', `Error destroying emulation: ${err.message}`);
    }
  }
}
