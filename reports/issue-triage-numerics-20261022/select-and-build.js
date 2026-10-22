// Selects the test issues where the first comparison's Doppler and Transformers.js
// runs chose different raw routes, plus a deterministic control sample where they
// agreed, and rebuilds each route prompt exactly as the application does
// (budgets 1600/800/300 characters, prompt must stay below maxSeqLen tokens).
// Writes prompts to a local file; issue text is never committed.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
const [repoRoot, csvDirectory, outputPath, controlCount = '16'] = process.argv.slice(2);
if (!outputPath) throw new Error('usage: select-and-build.js <repoRoot> <csvDir> <prompts.json> [controlCount]');
const app = path.join(repoRoot, 'examples/issue-triage');
const { loadSplit } = await import(path.join(app, 'dataset.js'));
const { TRIAGE_POLICY, buildRouteRequest } = await import(path.join(app, 'triage.js'));
const { AutoTokenizer, env } = await import(path.join(repoRoot, 'node_modules/@huggingface/transformers/dist/transformers.node.mjs'));
const runs = path.join(repoRoot, 'reports/issue-triage-comparison-20261022/runs');
const doppler = JSON.parse(await fs.readFile(path.join(runs, 'doppler-0.json'), 'utf8')).quality.test;
const tjs = JSON.parse(await fs.readFile(path.join(runs, 'transformersjs-0.json'), 'utf8')).quality.test;
const tjsByIndex = new Map(tjs.map((r) => [r.index, r]));
const disagree = doppler.filter((r) => tjsByIndex.get(r.index).suggestedRoute !== r.suggestedRoute);
const key = (r) => createHash('sha256').update(`numerics-control-20261022:${r.index}`).digest('hex');
const control = doppler.filter((r) => tjsByIndex.get(r.index).suggestedRoute === r.suggestedRoute)
  .sort((a, b) => (key(a) < key(b) ? -1 : 1)).slice(0, Number(controlCount));
const rows = await loadSplit('test', { readLocal: (split) => fs.readFile(path.join(csvDirectory, `issues_${split}.csv`), 'utf8') });
const byIndex = new Map(rows.map((r) => [r.index, r]));
env.allowRemoteModels = false;
env.localModelPath = process.env.TOKENIZER_ROOT;
const tokenizer = await AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-0.6B');
const count = (text) => tokenizer(text, { add_special_tokens: false }).input_ids.data.length;
const issues = [];
for (const [group, list] of [['disagree', disagree], ['control', control]]) {
  for (const r of list) {
    const issue = byIndex.get(r.index);
    let budget = null;
    for (const b of TRIAGE_POLICY.bodyCharacterBudgets) {
      if (TRIAGE_POLICY.routes.every((route) => count(buildRouteRequest(TRIAGE_POLICY, route, issue, b).prompt) < TRIAGE_POLICY.maxSeqLen)) { budget = b; break; }
    }
    issues.push({ index: r.index, group, label: r.label, bodyCharacterBudget: budget,
      routes: TRIAGE_POLICY.routes.map((route) => ({ id: route.id, prompt: buildRouteRequest(TRIAGE_POLICY, route, issue, budget).prompt })) });
  }
}
await fs.writeFile(outputPath, `${JSON.stringify({ policy: TRIAGE_POLICY.schema, issues }, null, 1)}\n`);
console.log(JSON.stringify({ disagree: disagree.length, control: control.length }));
