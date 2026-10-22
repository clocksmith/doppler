// Full-precision oracle: Transformers.js (onnxruntime-node, CPU) on the FP32 ONNX
// export of Qwen/Qwen3-Reranker-0.6B@e61197ed. Scores each route prompt with the
// application's yes/no logits at the answer position.
import fs from 'node:fs/promises';
import path from 'node:path';
const [repoRoot, modelRoot, promptsPath, outputPath] = process.argv.slice(2);
if (!outputPath) throw new Error('usage: fp32-oracle.js <repoRoot> <modelRoot> <prompts.json> <out.json>');
const { AutoTokenizer, AutoModelForCausalLM, env } = await import(path.join(repoRoot, 'node_modules/@huggingface/transformers/dist/transformers.node.mjs'));
env.allowRemoteModels = false;
env.localModelPath = modelRoot;
const id = 'Qwen/Qwen3-Reranker-0.6B';
const tokenizer = await AutoTokenizer.from_pretrained(id);
const model = await AutoModelForCausalLM.from_pretrained(id, { device: 'cpu', dtype: 'fp32' });
const encode = (text) => Array.from(tokenizer(text, { add_special_tokens: false }).input_ids.data, Number);
const { issues } = JSON.parse(await fs.readFile(promptsPath, 'utf8'));
const out = [];
for (const issue of issues) {
  const routes = [];
  for (const route of issue.routes) {
    const promptIds = encode(route.prompt);
    const [yes, no] = ['yes', 'no'].map((label) => encode(route.prompt + label).at(-1));
    const output = await model(tokenizer(route.prompt, { add_special_tokens: false }));
    const { data, dims } = output.logits;
    const offset = (dims.at(-2) - 1) * dims.at(-1);
    routes.push({ id: route.id, promptTokens: promptIds.length, yes: data[offset + yes], no: data[offset + no], score: data[offset + yes] - data[offset + no] });
  }
  out.push({ index: issue.index, group: issue.group, label: issue.label, bodyCharacterBudget: issue.bodyCharacterBudget, routes,
    suggestedRoute: routes.reduce((a, b) => (b.score > a.score ? b : a)).id });
  process.stdout.write('.');
}
await fs.writeFile(outputPath, `${JSON.stringify({ implementation: 'Transformers.js 4.2.0 onnxruntime-node CPU, FP32 ONNX (model.onnx 7c2db33a..., model.onnx_data 9ce4eefb...)', issues: out }, null, 1)}\n`);
