// Write the GPU run files for the routing-aware fresh test from one resolved runtime profile.
//
// Usage (from the runtime snapshot root):
//   node reports/flash-next-routing-aware-fresh-20261112/prepare-runs.js PLAN.json
// PLAN: { root, modelDir, expectedManifestSha256, runtimeRevision, policySha256, promptSetSha256,
//         sameInputCase, groups: { <group>: [caseId, ...] } }
// Inputs must already be frozen at ROOT/frozen/inputs.json; CPU references are expected at
// ROOT/cpu/<group>/round-0/learned-forward-reference-float32-artifact.json. Starts no GPU work.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { applyRuntimeProfile } from '../../src/tooling/browser-harness/runtime-config.js';
import { samplePositions } from '../../tools/lib/routing-aware-acceptance.js';

const PROFILE = 'profiles/flash-next-routed-f32kv-qualification';
const snapshot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const [planPath] = process.argv.slice(2);
assert.ok(planPath, 'Usage: prepare-runs.js PLAN.json');
const plan = JSON.parse(await fs.readFile(planPath));
const root = path.resolve(plan.root);
const config = path.join(root, 'config');
await fs.mkdir(config);
installNodeFileFetchShim();
const runtimeConfig = await applyRuntimeProfile(PROFILE);
assert.equal(runtimeConfig.inference.kernelPath ?? null, null, 'the named profile must not carry an inline execution graph');
assert.equal(runtimeConfig.inference.session.kvcache.kvDtype, 'f32');
const write = async (name, value) => {
  const file = path.join(config, name);
  const bytes = Buffer.from(JSON.stringify(value, null, 2) + '\n');
  await fs.writeFile(file, bytes, { flag: 'wx' });
  return { file, sha256: sha256(bytes) };
};
await write('runtime-config.json', { runtimeProfile: PROFILE, runtimeConfig });
const inputsPath = path.join(root, 'frozen', 'inputs.json');
const inputs = JSON.parse(await fs.readFile(inputsPath));
// plan.promptSetPath exists only for harness smoke tests on non-pinned prompts.
const promptSetPath = plan.promptSetPath
  ? path.resolve(plan.promptSetPath)
  : path.join(snapshot, 'tools', 'data', 'flash-next-routing-acceptance-prompts-v1.json');
const promptSet = JSON.parse(await fs.readFile(promptSetPath));
const allCases = Object.values(plan.groups).flat();
assert.deepEqual([...allCases].sort(), promptSet.cases.map(entry => entry.id).sort(), 'groups must cover every pinned case once');
const common = { modelDir: plan.modelDir, expectedManifestSha256: plan.expectedManifestSha256, runtimeConfig,
  runtimeProfile: PROFILE, runtimeRevision: plan.runtimeRevision };

await write('task-quality-run.json', { schema: 'doppler.flash-next-task-quality-run/v1', ...common, inputsPath,
  promptSetPath, outputDirectory: path.join(root, 'gpu', 'task-quality'), sequences: allCases,
  policySha256: plan.policySha256, promptSetSha256: plan.promptSetSha256 });

const sameInput = inputs.sequences.find(row => row.id === plan.sameInputCase);
assert.ok(sameInput, 'same-input case must be pinned');
await write('same-input-run.json', { schema: 'doppler.flash-next-same-input-capture-run/v1', ...common, inputsPath,
  sequenceId: plan.sameInputCase, positions: samplePositions(promptSet.samplePositions, sameInput.tokens.length),
  outputDirectory: path.join(root, 'same-input', 'capture'), policySha256: plan.policySha256 });

for (const [group, cases] of Object.entries(plan.groups)) {
  const referencePath = path.join(root, 'cpu', group, 'round-0', 'learned-forward-reference-float32-artifact.json');
  // Plain learned-forward run: no routing or stage instrumentation (observation neutrality).
  const plain = await write(`plain-${group}.json`, { schema: 'doppler.flash-next-learned-probe/v1', ...common,
    referencePath, inputsPath, outputPath: path.join(root, 'gpu', 'plain', group, 'observation.json'),
    sequences: cases.map(id => ({ id, maxTokens: 1 })), captureLayers: false, lifecycle: false });
  // Unforced observe and forced-indices runs, one model load (run-routing-controls.js).
  await write(`routing-controls-${group}.json`, { root, plainRunPath: plain.file, plainRunSha256: plain.sha256,
    referencePath, inputsPath, runtimeRevision: plan.runtimeRevision, cases, modes: ['observe', 'indices'] });
}
console.log(JSON.stringify({ config, profile: PROFILE, cases: allCases.length }));
