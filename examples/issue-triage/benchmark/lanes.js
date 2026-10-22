// Benchmark lanes. Each lane opens one engine and answers the same three-way
// routing question with probabilities over bug, feature and question.
//   open(config) -> { decide(issue) -> { probabilities, choice, detail }, runtime, close() }
// Reranker lanes score each route with the application's own prompts and turn
// route scores (yes - no) into probabilities with a temperature-1 softmax.
import { TRIAGE_POLICY, documentText, suggestRoute } from '../triage.js';

export const ROUTES = TRIAGE_POLICY.routes.map((route) => route.id);
export const OPEN_JEV_QUESTION = Object.freeze({
  instructions: 'What type of GitHub issue is this?',
  options: ROUTES,
  descriptions: Object.fromEntries(TRIAGE_POLICY.routes.map((route) => [route.id, route.query])),
});

export function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function rerankerDecider(scoreChoices) {
  return async (issue) => {
    const suggestion = await suggestRoute(scoreChoices, issue);
    const scores = ROUTES.map((id) => suggestion.routes.find((route) => route.id === id).score);
    const p = softmax(scores);
    return {
      probabilities: Object.fromEntries(ROUTES.map((id, i) => [id, p[i]])),
      choice: suggestion.suggestedRoute,
      detail: { routes: suggestion.routes, bodyCharacterBudget: suggestion.bodyCharacterBudget },
    };
  };
}

async function configureTransformers(transformers, hfBase) {
  Object.assign(transformers.env, {
    allowLocalModels: false,
    allowRemoteModels: true,
    useBrowserCache: true,
    remoteHost: hfBase,
  });
  // Serve ONNX Runtime's WebAssembly from the same origin instead of its CDN default.
  transformers.env.backends.onnx.wasm.wasmPaths = new URL('/app/benchmark/node_modules/onnxruntime-web/dist/', location.href).href;
}

export const LANES = {
  'doppler-capsule-q4k': {
    comparison: 'b-product',
    async open({ descriptor }) {
      const { openDopplerEngine } = await import('../engines.js');
      const engine = await openDopplerEngine({ descriptor, baseUrl: location.href, mode: 'install' });
      return { decide: rerankerDecider(engine.scoreChoices), runtime: { engine: 'doppler', surface: 'capsule', observations: engine.observations.length }, close: engine.close };
    },
  },
  'doppler-f16': {
    comparison: 'a-runtime-parity',
    async open() {
      globalThis.__DOPPLER_KERNEL_BASE_PATH__ = '/app/benchmark/node_modules/doppler-gpu/src/gpu/kernels/';
      const { load } = await import('doppler-gpu/compat');
      const model = await load({ url: new URL('/models/doppler-f16/', location.href).href }, {});
      const stats = model.advanced?.getStats?.() ?? {};
      return {
        decide: rerankerDecider((request, options) => model.scoreChoices(request, options)),
        runtime: { engine: 'doppler', surface: 'compat', weights: 'f16', activations: 'f32', adapterInfo: stats.adapterInfo ?? null },
        close: () => model.unload(),
      };
    },
  },
  'transformersjs-fp16': {
    comparison: 'a-runtime-parity',
    async open({ hfBase }) {
      const transformers = await import('@huggingface/transformers');
      await configureTransformers(transformers, hfBase);
      const { openTransformersEngine } = await import('../engines.js');
      const engine = await openTransformersEngine({ transformers, modelId: 'Qwen/Qwen3-Reranker-0.6B', dtype: 'fp16' });
      return { decide: rerankerDecider(engine.scoreChoices), runtime: { engine: 'transformersjs', version: transformers.env.version, device: 'webgpu', dtype: 'fp16' }, close: engine.close };
    },
  },
  ...Object.fromEntries([['open-jev-kev-0.6b', 'kev-0.6b'], ['open-jev-kev-4b', 'kev-4b'], ['open-jev-deberta', 'open-jev']].map(([lane, model]) => [lane, {
    comparison: 'b-product',
    async open({ hfBase }) {
      const transformers = await import('@huggingface/transformers');
      await configureTransformers(transformers, hfBase);
      const { OpenJev, choice } = await import('open-jev');
      const jev = await OpenJev.load({ model });
      const question = choice(OPEN_JEV_QUESTION.instructions, OPEN_JEV_QUESTION.options, OPEN_JEV_QUESTION.descriptions);
      return {
        async decide(issue) {
          const [answer] = await jev.decide(documentText(issue, TRIAGE_POLICY.bodyCharacterBudgets[0]), [question]);
          return { probabilities: { ...answer.probabilities }, choice: answer.choice, detail: { confidence: answer.confidence } };
        },
        runtime: { engine: 'open-jev', version: '0.1.2', transformersVersion: transformers.env.version, ...jev.runtime },
        close: () => jev.dispose(),
      };
    },
  }])),
};
