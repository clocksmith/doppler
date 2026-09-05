import fs from 'node:fs/promises';
import path from 'node:path';
import { bootstrapNodeWebGPU } from '../src/tooling/node-webgpu.js';
import { createForecastProgramFactory } from '../src/inference/pipelines/forecast/pack-program.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { openPack } from '../src/pack-runtime.js';

const root = path.resolve(process.argv[2] ?? '');
const sealed = process.argv.includes('--sealed');
const candidate = JSON.parse(await fs.readFile(path.join(root, 'candidate.json'), 'utf8'));
const reference = JSON.parse(await fs.readFile(path.join(root, 'source/reference.json'), 'utf8'));
const artifactStore = { async readArtifact(artifact) {
  const bytes = await fs.readFile(path.join(root, artifact.path));
  if (hashBytesSha256(bytes) !== artifact.hash || bytes.length !== artifact.sizeBytes) throw new Error('Qualification artifact custody mismatch: ' + artifact.path);
  return new Uint8Array(bytes);
} };
const started = new Date().toISOString();
const rows = [];
let failure = null;
let device;
let program;
let bootstrap;
let adapterInfo;
let profile;
let checkpoint = { sequence: 0, digest: null };
let pack;
try {
  bootstrap = await bootstrapNodeWebGPU();
  if (!bootstrap.ok) throw new Error(bootstrap.detail);
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter.');
  adapterInfo = Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(k => [k, adapter.info[k] ?? null]));
  device = await adapter.requestDevice();
  if (sealed) {
    pack = JSON.parse(await fs.readFile(path.join(root, 'pack.json'), 'utf8'));
    profile = JSON.parse(await fs.readFile(path.join(root, 'trust-profile.json'), 'utf8'));
    const releaseEvents = JSON.parse(await fs.readFile(path.join(root, 'release-events.json'), 'utf8'));
    program = await openPack(pack, { artifactStore, trustedSigners: profile.trustedSigners,
      device: { getDevice: () => device, getProfile: () => ({ surface: 'node-webgpu', hasF16: device.features.has('shader-f16'),
        hasSubgroups: device.features.has('subgroups'), maxBufferSize: device.limits.maxBufferSize }) },
      programFactory: createForecastProgramFactory(device), session: { releaseEvents,
        releaseTrustedSigners: profile.trustedSigners, acceptedTargetPlanDigests: profile.acceptedTargetPlanDigests,
        releasePolicy: { now: new Date().toISOString(), minimumSequence: profile.minimumSequence, checkpoint },
        persistReleaseCheckpoint: async value => {
          await fs.writeFile(path.join(root, 'node-release-checkpoint.json'), JSON.stringify(value)); checkpoint = value;
        } } });
  } else program = await createForecastProgramFactory(device)({ pack: candidate, targetPlan: candidate.targetPlan, artifactStore });
  for (const testCase of reference.cases) {
    const start = performance.now();
    const request = { context: testCase.context, horizon: testCase.horizon };
    const output = sealed ? await program.forecast({ ...request, application: profile.application, assignmentHash: null })
      : await program.forecast(request, { signal: undefined });
    let maxAbsoluteError = 0;
    let mismatches = 0;
    output.values.forEach((value, i) => {
      const error = Math.abs(value - testCase.values[i]);
      maxAbsoluteError = Math.max(maxAbsoluteError, error);
      if (error > reference.tolerance.absolute + reference.tolerance.relative * Math.abs(testCase.values[i])) mismatches++;
    });
    const row = { id: testCase.id, elapsedMs: performance.now() - start, maxAbsoluteError, mismatches, output };
    rows.push(row);
    console.log(JSON.stringify({ id: row.id, elapsedMs: row.elapsedMs, maxAbsoluteError, mismatches }));
  }
  if (rows.some(row => row.mismatches > 0)) throw new Error('Physical forecasting reference parity failed.');
} catch (error) {
  failure = String(error?.stack ?? error);
  console.error(failure);
  process.exitCode = 1;
} finally {
  try { await program?.close(); } finally { device?.destroy(); await bootstrap?.session?.close(); }
  const report = { schema: 'doppler.forecast-qualification/v1', startedAt: started, completedAt: new Date().toISOString(),
    boundary: sealed ? 'signed-pack-session' : 'candidate-program', packIdentity: program?.packIdentity ?? null, checkpoint,
    status: failure ? 'failed' : 'passed', surface: 'node-webgpu', adapter: adapterInfo ?? null,
    provider: bootstrap?.provider ?? null, node: process.version,
    candidateHash: computeCanonicalSha256(candidate), executionGraphHash: candidate.program.executionGraphHash,
    artifactClosureHash: computeCanonicalSha256(candidate.artifacts), referenceHash: computeCanonicalSha256(reference),
    tolerance: reference.tolerance, rows, failure };
  const filename = `qualification-${sealed ? 'signed-' : ''}node-${started.replace(/[:.]/g, '-')}.json`;
  await fs.writeFile(path.join(root, filename), JSON.stringify(report));
  console.log(JSON.stringify({ report: path.join(root, filename), status: report.status }));
}
