import { encodeSharedPrompt } from './shared-input.js';
import { measureDopplerRequest, measurePooledRequest } from './resident-request.js';

try {
  const config = await (await fetch('/probe-config.json')).json();
  const { contract, kind } = config;
  let session, pipeline, runtimeConfig, runTextInference, device, tokenizer, identity;
  const gpuErrors = [];
  if (kind === 'pooled') {
    const { initializePooledSession } = await import('./pooled-session.js');
    session = await initializePooledSession(config, { observeUploads: true });
    const { verifyPooledGpu } = await import('./verify-pooled-gpu.js');
    const materialization = await verifyPooledGpu(session.device, session.weights, session.observation.uploaded, contract.materializationObservation);
    ({ device, tokenizer } = session);
    identity = { materialization, adapter: session.adapter.info, settings: { kvDtype: session.kvDtype,
      attnFlash: session.engine.flash, maxSeq: session.engine.maxSeq }, input: session.weights.canonicalProof };
  } else if (kind === 'doppler') {
    const { initializeSuiteModel } = await import('/runtime/src/tooling/browser-harness/model-initialization.js');
    ({ runTextInference } = await import('/runtime/src/tooling/browser-harness/text-execution.js'));
    const { getRuntimeConfig } = await import('/runtime/src/config/runtime.js');
    const { getDevice } = await import('/runtime/src/gpu/device.js');
    const harness = await initializeSuiteModel({ ...config.doppler, modelUrl: new URL('/rdrr', location.href).href,
      runtime: { runtimeConfig: config.doppler.runtimeConfig } });
    pipeline = harness.pipeline;
    device = getDevice();
    tokenizer = pipeline.tokenizer;
    runtimeConfig = getRuntimeConfig();
    const { readBuffer } = await import('/runtime/src/memory/buffer-pool.js');
    const { verifyDopplerGpu } = await import('./verify-doppler-gpu.js');
    const canonical = await (await fetch('/canonical/canonical.json')).json();
    const materialization = await verifyDopplerGpu(pipeline, canonical, readBuffer);
    const reference = runtimeConfig.shared?.harness?.referenceTranscript;
    if (runtimeConfig.shared?.debug?.probes?.length || runtimeConfig.shared?.debug?.profiler?.enabled
        || runtimeConfig.shared?.benchmark?.run?.executionObserver?.enabled
        || reference?.enabled || reference?.captureLogits || reference?.captureKvBytes
        || runtimeConfig.shared?.harness?.mode === 'diagnose') throw new Error('Timed runtime enables diagnostic instrumentation');
    identity = { materialization, capabilities: harness.capabilities, manifest: harness.manifest, runtimeConfig,
      execution: (await import('/runtime/src/config/initial-execution-identity.js')).observeInitialExecutionIdentity(pipeline.resolvedRuntimeSession) };
    const actualStops = [...new Set([...pipeline.modelConfig.stopTokenIds,
      tokenizer.getSpecialTokens?.()?.eos].filter(Number.isSafeInteger))].sort((a, b) => a - b);
    if (JSON.stringify(actualStops) !== JSON.stringify([...contract.shared.eosTokenIds].sort((a, b) => a - b))) {
      throw new Error('Doppler effective EOS rules differ from shared contract');
    }
  } else throw new Error('Unknown resident engine realm');
  device.addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
  const promptIds = {};
  for (const prompt of config.prompts) {
    promptIds[prompt.id] = kind === 'pooled' ? encodeSharedPrompt(tokenizer, prompt.text, contract.promptFormat)
      : Array.from(tokenizer.encode(contract.promptFormat.before + prompt.text + contract.promptFormat.after));
    if (JSON.stringify(promptIds[prompt.id]) !== JSON.stringify(prompt.promptIds)) throw new Error(`Prompt encoding mismatch: ${prompt.id}`);
  }
  const adapter = kind === 'pooled' ? session.adapter.info : identity.capabilities.adapterInfo;
  if (adapter.vendor.toLowerCase() !== config.requiredVendor) throw new Error('Unexpected physical GPU vendor');
  await device.queue.onSubmittedWorkDone();
  globalThis.RESIDENT = {
    async prepare(prompt) {
      if (gpuErrors.length || session?.gpuErrors.length) throw new Error('Prior GPU validation failure');
      if (kind === 'pooled') session.engine.reset();
      else pipeline.reset();
      await device.queue.onSubmittedWorkDone();
      if (JSON.stringify(promptIds[prompt.id]) !== JSON.stringify(prompt.promptIds)) throw new Error('Prepared prompt identity changed');
    },
    async request(prompt) {
      let result;
      try {
        result = kind === 'pooled'
          ? await measurePooledRequest({ ...session, prompt, contract })
          : await measureDopplerRequest({ pipeline, device, runTextInference, runtimeConfig, prompt, contract, method: config.method });
        if (kind === 'doppler') {
          const stats = pipeline.getStats(), memory = pipeline.getMemoryStats();
          result.execution = { plan: stats.executionPlan, kvDtype: memory.kvCache?.kvDtype,
            activationDtype: memory.decodeBuffers?.activationDtype, decodeMode: stats.decodeMode };
          if (result.execution.kvDtype !== contract.shared.kvDtype
              || result.execution.activationDtype !== contract.shared.activationDtype
              || stats.executionPlan?.transitions?.length
              || stats.executionPlan?.finalActivePlanId !== stats.executionPlan?.activePlanIdAtStart) throw new Error('Actual runtime dtype/fallback differs from contract');
        }
        if (gpuErrors.length || session?.gpuErrors.length) throw new Error(`GPU validation failure: ${[...gpuErrors, ...(session?.gpuErrors ?? [])].join('; ')}`);
        return { result };
      } catch (error) {
        return { error: String(error.stack || error), partialRequest: result ?? error.partialRequest ?? null };
      }
    },
    async close() {
      await device.queue.onSubmittedWorkDone();
      if (kind === 'pooled') session.engine.destroy?.();
      else await pipeline.unload?.();
      device.destroy();
    },
  };
  globalThis.READY = { kind, identity, promptIds, gpuErrors, adapter: {
    vendor: adapter.vendor, architecture: adapter.architecture, device: adapter.device,
    description: adapter.description, subgroupMinSize: adapter.subgroupMinSize, subgroupMaxSize: adapter.subgroupMaxSize,
  }, deviceLimits: Object.fromEntries(['maxBufferSize', 'maxStorageBufferBindingSize',
    'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY',
    'maxComputeWorkgroupSizeZ', 'maxComputeWorkgroupStorageSize', 'maxComputeWorkgroupsPerDimension']
    .map(key => [key, device.limits[key]])),
    features: [...device.features], userAgent: navigator.userAgent };
} catch (error) {
  globalThis.READY = { error: String(error.stack || error), completedChecks: error.completedChecks ?? null };
}
