// In-page driver for tools/product-compare.js. Uses the application's own
// policy and engine adapters; only the engine differs between runs.
import { suggestRoute } from '/app/triage.js';
import { openDopplerEngine, openTransformersEngine } from '/app/engines.js';

let engine = null;
window.harness = {
  async open(engineName, config) {
    const started = performance.now();
    if (engineName === 'doppler') {
      engine = await openDopplerEngine({ descriptor: config.descriptor, baseUrl: location.href, mode: config.mode });
    } else {
      const transformers = await import('/tjs-package/dist/transformers.web.js');
      Object.assign(transformers.env, {
        allowLocalModels: false,
        allowRemoteModels: true,
        useBrowserCache: true,
        remoteHost: new URL('/tjs-remote/', location.href).href,
      });
      engine = await openTransformersEngine({ transformers, modelId: config.modelId });
      engine.version = transformers.env.version;
    }
    return { openMs: performance.now() - started, observations: engine.observations.length, version: engine.version ?? null };
  },
  async triage(issue) {
    const started = performance.now();
    const suggestion = await suggestRoute(engine.scoreChoices, issue);
    return { ...suggestion, elapsedMs: performance.now() - started };
  },
  async cancelProbe(issue, delayMs) {
    const controller = new AbortController();
    const started = performance.now();
    let abortedAt = null;
    const settled = suggestRoute(engine.scoreChoices, issue, { signal: controller.signal })
      .then(() => ({ status: 'completed' }), (error) => ({ status: error.name, code: error.code ?? null, message: String(error.message).slice(0, 200) }));
    setTimeout(() => { abortedAt = performance.now(); controller.abort(); }, delayMs);
    const result = await settled;
    const settledAt = performance.now();
    return { ...result, delayMs, totalMs: settledAt - started, abortToSettleMs: abortedAt == null ? null : settledAt - abortedAt };
  },
  async close() {
    await engine?.close();
    engine = null;
  },
};
window.harnessReady = true;
