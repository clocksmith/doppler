import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs/promises';
import path from 'node:path';
import { hashFile, hashJson } from './resident-snapshot.js';

const exec = promisify(execFile);
const equal = (a, b, label) => { if (hashJson(a) !== hashJson(b)) throw new Error(`Proof identity mismatch: ${label}`); };
const rows = files => files.map(({ path, sha256 }) => ({ path, sha256 })).sort((a, b) => a.path.localeCompare(b.path, 'en'));
const sorted = value => Array.isArray(value) ? value.map(sorted) : value && typeof value === 'object'
  ? Object.fromEntries(Object.keys(value).sort().map(key => [key, sorted(value[key])])) : value;
export function assertKnownAdapter(left, right) {
  if (['vendor', 'architecture'].some(key => !left[key] || left[key] !== right[key])
      || (left.device && right.device && left.device !== right.device)) throw new Error('Known physical adapter fields differ');
}
function inferenceIdentity(runtime) {
  const inference = structuredClone(runtime?.inference);
  if (!inference) throw new Error('Proof lacks inference configuration');
  delete inference.prompt;
  return sorted(inference);
}
// Numerical eligibility is reused by exact source and inference identity.
// Resident materialization is verified separately on each timed instance.
export async function admitResidentProofs(config, identity, output) {
  const retained = [];
  async function read(reference, label) {
    if (!reference?.path || !/^[a-f0-9]{64}$/.test(reference.sha256)) throw new Error(`Pin ${label} receipt path and SHA256`);
    if (await hashFile(reference.path) !== reference.sha256) throw new Error(`Receipt drift: ${label}`);
    const bytes = await fs.readFile(reference.path);
    await fs.writeFile(path.join(output, `${label}.json`), bytes, { flag: 'wx' });
    retained.push({ label, sha256: reference.sha256 });
    return JSON.parse(bytes);
  }
  await fs.mkdir(output);
  const binding = await read(config.proofBinding, 'binding');
  if (binding.schema !== 'doppler.equal-weight-resident-numerical-binding.v1') throw new Error('Unsupported numerical binding packet');
  const canonicalPath = path.join(config.canonicalDir, 'canonical.json');
  const canonicalReference = { path: canonicalPath, sha256: await hashFile(canonicalPath) };
  const canonical = await read(canonicalReference, 'canonical');
  if (canonical.schema !== 'doppler.equal-weight-canonical.v1' || !canonical.completed
      || canonical.sourceManifestSha256 !== config.contract.sourceManifestSha256
      || canonical.tensors.length !== config.method.expectedTensorCount) throw new Error('Incomplete canonical inventory');
  const canonicalFile = identity.artifacts.canonical.find(file => file.path === 'canonical.json');
  if (canonicalFile?.sha256 !== canonicalReference.sha256) throw new Error('Current canonical inventory differs');
  const source = await read(binding.dopplerSource, 'doppler-source');
  equal(rows(source.files.filter(file => file.path.startsWith('src/') && file.symlink === undefined)),
    rows(identity.runtimeFiles.filter(file => file.path.startsWith('src/') && file.symlink === undefined)), 'complete original Doppler runtime');
  if (!/^[a-f0-9]{40}$/.test(source.commit)) throw new Error('Pin numerical source commit');
  const originalTree = (await exec('git', ['-C', config.dopplerRoot, 'rev-parse', `${source.commit}:src`])).stdout.trim();
  equal(originalTree, identity.doppler.srcTree, 'complete runtime Git tree including aliases');
  const observation = await read(binding.dopplerObservation, 'doppler-observation');
  if (!observation.completed || observation.numericPrototype !== null) throw new Error('Invalid original observation boundary');
  const originalFiles = new Map(source.files.map(file => [file.path, file.sha256]));
  for (const [file, sha] of [[observation.file, observation.originalSha256],
    ['src/inference/pipelines/text/probes.js', observation.probeOriginalSha256],
    ['src/inference/pipelines/text/generator.js', observation.logitObservation?.originalSha256],
    ['src/inference/pipelines/text/generator-steps.js', observation.logitObservation?.originalStepsSha256],
    ['src/inference/pipelines/text/generator/decode.js', observation.logitObservation?.originalDecodeSha256]]) {
    if (!sha || originalFiles.get(file) !== sha) throw new Error(`Observer original-source binding failed: ${file}`);
  }
  const numerical = await read(binding.numerical, 'numerical');
  const subgroup = numerical.schema === 'doppler.subgroup-f32-numerical-comparison.v1';
  if (!subgroup && numerical.schema !== 'doppler.pooled-f32kv-numerical-comparison.v1') throw new Error('Unsupported numerical producer schema');
  const cases = subgroup ? numerical.cases : numerical.prompts;
  if (numerical.pass !== true || cases?.length !== config.prompts.length) throw new Error('Four-prompt numerical gate required');
  let pooledNumerical;
  if (subgroup) {
    equal(numerical.sourceCommit, source.commit, 'subgroup numerical source commit');
    pooledNumerical = await read(binding.pooledNumerical, 'pooled-numerical');
    if (pooledNumerical.schema !== 'doppler.pooled-f32kv-numerical-comparison.v1') throw new Error('Pinned Pooled-native teacher evidence required');
    equal(numerical.teacherReceiptSha256, pooledNumerical.teacherReceiptSha256, 'shared independent teacher');
  }
  function coverage(checks, label, select) {
    const values = checks.filter(select);
    if (values.some(check => check.pass !== true || !Number.isFinite(check.maxAbs) || check.maxAbs < 0
        || check.maxAbs > config.contract.gates.logitAbsoluteTolerance
        || (check.failedValues !== undefined && check.failedValues !== 0))) throw new Error(`Numerical values failed: ${label}`);
    equal(values.map(check => check.step).sort((a, b) => a - b), config.contract.gates.logitPositions, `${label} full logit positions`);
  }
  let adapter, browserUserAgent;
  for (const prompt of config.prompts) {
    const row = cases.find(value => value.id === prompt.id);
    if (!row || row.promptIdsEqual !== true
        || (subgroup ? row.tokensEqual !== true || row.tokens !== config.contract.shared.maxNewTokens || row.pass !== true
          : row.tokenIdsEqual !== true || row.tokenCount !== config.contract.shared.maxNewTokens)) throw new Error(`Numerical gate failed/incomplete: ${prompt.id}`);
    if (subgroup) {
      coverage(row.checks, `${prompt.id} doppler-native`, check => check.against === 'native');
      coverage(row.checks, `${prompt.id} doppler-pooled`, check => check.against === 'pooled');
      const pooledCase = pooledNumerical.prompts.find(value => value.id === prompt.id);
      if (!pooledCase || pooledCase.pooledSha256 !== row.pooledReceiptSha256) throw new Error('Pooled numerical receipt changed');
      coverage(pooledCase.checks, `${prompt.id} pooled-native`, check => check.comparison === 'pooled-native');
    } else {
      for (const comparison of ['doppler-native', 'pooled-native', 'doppler-pooled']) {
        coverage(row.checks, `${prompt.id} ${comparison}`, check => check.comparison === comparison);
      }
    }
    const doppler = await read(binding.prompts[prompt.id].doppler, `${prompt.id}-doppler`);
    const pooled = await read(binding.prompts[prompt.id].pooled, `${prompt.id}-pooled`);
    equal(subgroup ? row.rawReceiptSha256 : row.dopplerSha256, binding.prompts[prompt.id].doppler.sha256, 'numerical Doppler receipt');
    equal(subgroup ? row.pooledReceiptSha256 : row.pooledSha256, binding.prompts[prompt.id].pooled.sha256, 'numerical Pooled receipt');
    equal(inferenceIdentity(doppler.request.runtimeConfig), inferenceIdentity(config.doppler.runtimeConfig), `${prompt.id} inference overlay`);
    if (doppler.ok !== true || doppler.result.failed !== 0) throw new Error('Doppler numerical request failed');
    const info = doppler.result.deviceInfo.adapterInfo;
    const currentAdapter = { vendor: info.vendor, architecture: info.architecture, device: info.device };
    const vendorAdapter = { vendor: pooled.adapter.vendor, architecture: pooled.adapter.architecture, device: pooled.adapter.device };
    assertKnownAdapter(currentAdapter, vendorAdapter);
    if (adapter) assertKnownAdapter(currentAdapter, adapter);
    adapter = currentAdapter;
    if (browserUserAgent) equal(browserUserAgent, doppler.result.env.browserUserAgent, 'numerical browser identity');
    browserUserAgent = doppler.result.env.browserUserAgent;
    const transcript = doppler.result.metrics.referenceTranscript;
    equal(transcript.prompt.ids, prompt.promptIds, `${prompt.id} prompt IDs`);
    equal(transcript.tokens.ids, prompt.expectedTokenIds, `${prompt.id} generated IDs`);
    equal(pooled.promptIds, prompt.promptIds, `${prompt.id} Pooled prompt IDs`);
    equal(pooled.tokenIds, prompt.expectedTokenIds, `${prompt.id} Pooled generated IDs`);
    equal(doppler.result.metrics.generatedText, prompt.expectedText, `${prompt.id} Doppler response text`);
    equal(pooled.text, prompt.expectedText, `${prompt.id} Pooled response text`);
    if (pooled.engineVariant !== 'pooled-native-f32' || pooled.gpuErrors?.length || pooled.browserErrors?.length
        || pooled.settings.kvDtype !== 'f32' || pooled.settings.attnFlash !== false) throw new Error('Pooled proof is not native F32-KV');
    const vendorSource = await read(binding.prompts[prompt.id].pooledSource, `${prompt.id}-pooled-source`);
    if (vendorSource.vendorChanges?.length) throw new Error('Native Pooled proof contains vendor mutations');
    equal(rows(vendorSource.engineFiles), rows(identity.vendorFiles.map(file => ({ ...file, path: `vendor/${file.path}` }))), 'complete native Pooled source');
  }
  return { pass: true, retained, adapter, browserUserAgent, proofBindingSha256: config.proofBinding.sha256,
    contract: 'Exact original source/overlay binding and full four-prompt numerical checks; timed instances obtain fresh actual-resource proofs during resident setup.' };
}
