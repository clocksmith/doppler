import fs from 'node:fs/promises';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
installNodeFileFetchShim();
const root = new URL('./', import.meta.url);
const read = async path => JSON.parse(await fs.readFile(path));
const base = await read(new URL('packed-conversion.json', root));
const donor = await read('src/config/conversion/qwen3/qwen3.8-27b-text-q4k-ehaf16.json');
const graph = structuredClone(base.execution);
graph.kernels.packed_decode = donor.execution.kernels.q4_decode;
graph.kernels.packed_prefill = donor.execution.kernels.q4_widetile;
const projections = new Set(['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']);
for (const phase of ['decode', 'prefill']) {
  for (const step of graph[phase]) {
    if (projections.has(step[0])) step[1] = `packed_${phase}`;
  }
}
const manifest = await read('.cache/mlx-gemma270m-q4k/manifest.json');
const compiled = compileExecutionV1({
  manifestInference: {...manifest.inference, execution: graph},
  modelId: 'gemma270m-packed-candidate', numLayers: manifest.architecture.numLayers,
  headDim: manifest.architecture.headDim, weightDtype: 'q4k',
});
const policy = await read(new URL('packed-policy.json', root));
policy.claim = 'Candidate packed GPU execution, unchanged Q4K artifact bytes. Residency and correctness diagnostic; no default promotion.';
policy.engines.doppler.runtimeConfig.inference = {
  kernelPath: compiled.runtimeInferencePatch.kernelPath,
  compute: {activationDtype: base.session.compute.defaults.activationDtype},
  session: {compute: manifest.inference.session.compute,
    kvcache: {...manifest.inference.session.kvcache, maxSeqLen: 4096}},
};
await fs.writeFile(new URL('packed-candidate-graph.json', root), JSON.stringify(graph, null, 2));
await fs.writeFile(new URL('packed-candidate-policy.json', root), JSON.stringify(policy, null, 2));
