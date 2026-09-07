#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

// Replay existing measurement/application tools. The candidate evaluator owns
// selection and correctness; this entrypoint only binds portable paths and waits.
const [runtime, restoration, output] = process.argv.slice(2);
assert(runtime && restoration && output && process.argv.length === 5,
  'Usage: node tools/reproduce-retention-experiment.js <installed-runtime-bundle> <restoration> <new-output-directory>');
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const inputs = path.join(repo, 'reports/retention-experiment/20260907/inputs');
const bindings = { runtime: path.resolve(runtime), restoration: path.resolve(restoration),
  output: path.resolve(output), repo, inputs };
const recipe = JSON.parse(await fs.readFile(path.join(inputs, 'reproduction-configs.json'), 'utf8'));
assert.equal(recipe.schema, 'doppler.retention-reproduction-configs/v1');
function bind(value) {
  if (Array.isArray(value)) return value.map(bind);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, bind(item)]));
  if (typeof value !== 'string' || !value.startsWith('@')) return value;
  const [name, ...parts] = value.slice(1).split('/');
  assert(Object.hasOwn(bindings, name), `Unknown reproduction binding: ${name}`);
  const result = path.resolve(bindings[name], ...parts);
  assert(result === bindings[name] || result.startsWith(bindings[name] + path.sep), 'Reproduction path escapes its binding.');
  return result;
}
const configs = bind(recipe.configs);
await fs.mkdir(bindings.output);
const report = { schema: 'doppler.retention-reproduction-result/v1', passed: false,
  startedAtUtc: new Date().toISOString(), bindings, commands: [], defaultChanged: false, promotionAllowed: false };
async function run(name, tool) {
  const configPath = path.join(bindings.output, `${name}.json`);
  await fs.writeFile(configPath, JSON.stringify(configs[name], null, 2) + '\n', { flag: 'wx' });
  const logPath = path.join(bindings.output, `${name}.log`);
  const log = await fs.open(logPath, 'wx');
  const args = [path.join(repo, 'tools', tool), configPath];
  report.commands.push({ executable: process.execPath, args, log: logPath });
  console.log(JSON.stringify({ stage: name }));
  try {
    await new Promise((resolve, reject) => {
      const child = spawn(process.execPath, args, { cwd: repo, stdio: ['ignore', log.fd, log.fd] });
      child.once('error', reject);
      child.once('exit', (code, signal) => code === 0 ? resolve() : reject(new Error(`${name} failed: ${code ?? signal}`)));
    });
  } finally { await log.close(); }
}
try {
  const restored = JSON.parse(await fs.readFile(path.join(bindings.restoration, 'restoration.json'), 'utf8'));
  assert(restored.passed, 'Successful public-source reconstruction required.');
  await run('node', 'run-capsule-retention-experiment.js');
  const experiment = JSON.parse(await fs.readFile(path.join(configs.node.outputDir, 'experiment.json'), 'utf8'));
  assert(experiment.correctnessPassed && experiment.heldoutImprovement, 'Node held-out optimization gate failed.');
  const selected = JSON.parse(await fs.readFile(path.join(configs.node.outputDir,
    `candidate-${experiment.selectedCandidateHash.slice(7)}.json`), 'utf8'));
  assert.notEqual(selected.maxRetainedArtifactBytes, null, 'A bounded candidate must win before browser transfer.');
  configs['build-bounded'].loading.maxRetainedArtifactBytes = selected.maxRetainedArtifactBytes;
  report.selectedCandidateHash = experiment.selectedCandidateHash;
  await fs.writeFile(path.join(bindings.output, 'browser-transfer-selection.json'), JSON.stringify({
    parentCandidateHash: experiment.selectedCandidateHash, loading: configs['build-bounded'].loading,
    sourceApplication: 'installed-node-reranker', targetApplication: 'persistent-browser-document-search',
    promoted: false }, null, 2) + '\n', { flag: 'wx' });
  for (const name of ['unlimited', 'bounded']) await run(`build-${name}`, 'build-document-search-app.js');
  for (const name of ['unlimited', 'bounded']) await run(`qualify-${name}`, 'qualify-document-search.js');
  await run('compare', 'compare-document-search-retention.js');
  report.passed = true;
} catch (error) { report.error = { message: error.message, stack: error.stack }; process.exitCode = 1; }
finally {
  report.completedAtUtc = new Date().toISOString();
  await fs.writeFile(path.join(bindings.output, 'reproduction.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, output: bindings.output, error: report.error }));
