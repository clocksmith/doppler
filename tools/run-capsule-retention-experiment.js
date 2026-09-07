#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import { computeCanonicalSha256 as canonical, hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { runForgeCandidateEvaluation } from '../src/converter/forge-candidate-evaluation.js';

export function generateRetentionCandidates({ runtimeHash, capsuleHash, targetPlanDigest, retentionBytes }) {
  assert(Array.isArray(retentionBytes) && retentionBytes.includes(null) && retentionBytes.length > 1);
  assert.equal(new Set(retentionBytes).size, retentionBytes.length, 'Duplicate retention candidate.');
  for (const hash of [runtimeHash, capsuleHash, targetPlanDigest]) assert(/^sha256:[a-f0-9]{64}$/.test(hash));
  return retentionBytes.map(maxRetainedArtifactBytes => {
    assert(maxRetainedArtifactBytes === null || Number.isSafeInteger(maxRetainedArtifactBytes) && maxRetainedArtifactBytes >= 0);
    return { schema: 'doppler.capsule-retention-candidate/v1', runtimeHash, capsuleHash,
      targetPlanDigest, maxRetainedArtifactBytes };
  });
}

export function validateExperimentSplit(tuning, heldout) {
  assert(tuning.length && heldout.length, 'Both tuning and held-out inputs are required.');
  const ids = [...tuning, ...heldout].map(row => row.id);
  assert.equal(new Set(ids).size, ids.length, 'Case IDs must be unique across splits.');
  const inputs = [...tuning, ...heldout].map(row => canonical(row.reference.input));
  assert.equal(new Set(inputs).size, inputs.length, 'Tuning and held-out inputs must be disjoint.');
  const tolerances = canonical(tuning[0].reference.tolerances);
  for (const row of [...tuning, ...heldout]) {
    assert.equal(canonical(row.reference.tolerances), tolerances, 'Reference tolerance policy must remain unchanged.');
    assert.equal(canonical(row.reference.source.files), canonical(tuning[0].reference.source.files), 'Source weights must match across splits.');
    assert.equal(canonical(row.reference.scoringConfig), canonical(tuning[0].reference.scoringConfig), 'Scoring semantics must match across splits.');
  }
}

export async function runRetentionExperiment(config) {
  assert.equal(config.schema, 'doppler.capsule-retention-experiment/v1');
  assert.equal(process.platform, 'linux', 'The process RSS measurement uses Linux units.');
  assert(!process.versions.bun, 'This measurement adapter requires Node; Bun qualification is separate.');
  const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
  async function pinned(input) {
    const bytes = await fs.readFile(input.path);
    assert.equal(hashBytesSha256(bytes), input.digest, `Experiment input changed: ${input.path}`);
    return JSON.parse(bytes.toString('utf8'));
  }
  assert.equal(hashBytesSha256(await fs.readFile(config.worker.path)), config.worker.digest, 'Worker changed.');
  assert(Array.isArray(config.workerDependencies) && config.workerDependencies.length > 0, 'Pinned worker dependencies required.');
  async function verifyWorkerDependencies() {
    for (const input of config.workerDependencies) assert.equal(hashBytesSha256(await fs.readFile(input.path)), input.digest, 'Worker dependency changed.');
  }
  await verifyWorkerDependencies();
  const baseline = await pinned(config.baseline);
  assert(baseline.passed && baseline.cleanup.passed, 'Passing installed baseline required.');
  const capsuleBytes = await fs.readFile(path.join(config.capsuleRoot, 'distribution/capsule-v3.json'));
  const capsule = JSON.parse(capsuleBytes.toString('utf8'));
  const candidates = generateRetentionCandidates({ runtimeHash: `sha256:${baseline.installedPackage.sha256}`,
    capsuleHash: hashBytesSha256(capsuleBytes), targetPlanDigest: baseline.plan, retentionBytes: config.retentionBytes });
  const cases = {};
  for (const split of ['tuning', 'heldout']) {
    cases[split] = [];
    for (const row of config[split]) cases[split].push({ ...row, reference: await pinned(row.sourceReference) });
  }
  validateExperimentSplit(cases.tuning, cases.heldout);
  await fs.mkdir(config.outputDir);
  async function write(name, value) {
    const filename = path.join(config.outputDir, name);
    const bytes = Buffer.from(JSON.stringify(value, null, 2) + '\n');
    await fs.writeFile(filename, bytes, { flag: 'wx' });
    return { path: filename, digest: hashBytesSha256(bytes) };
  }
  await write('frozen-campaign.json', config);
  const candidateFiles = new Map();
  for (const candidate of candidates) candidateFiles.set(canonical(candidate), await write(`candidate-${canonical(candidate).slice(7)}.json`, candidate));
  const report = { schema: 'doppler.capsule-retention-experiment-result/v1', configDigest: canonical(config),
    hypothesis: config.hypothesis, candidates, startedAtUtc: new Date().toISOString(),
    memoryDefinition: 'Linux fresh-process peak RSS in bytes, including mapped/shared memory; not GPU-only or unique system memory.',
    application: 'installed-node-reranker', correctnessPassed: false, heldoutImprovement: false,
    promotionAllowed: false, claimAllowed: false, defaultChanged: false };
  async function evaluate(split, hashes) {
    const rows = cases[split];
    const output = source => ({ tokens: source.map(row => row.tokenIds),
      logits: source.flatMap(row => [row.trueLogit, row.falseLogit]), scores: source.map(row => row.score),
      probabilities: source.map(row => row.probability), budget: true });
    const reference = { schema: 'doppler.forge-source-reference/v1',
      sourceHash: canonical(rows[0].reference.source.files), oracleHash: canonical(rows.map(row => row.sourceReference.digest)),
      cases: rows.map(row => ({ id: row.id, input: row.reference.input, expected: output(row.reference.outputs) })) };
    const tolerance = rows[0].reference.tolerances;
    const contract = { schema: 'doppler.forge-candidate-evaluation/v1', evaluationId: `${config.id}-${split}`,
      modelIRHash: canonical(capsule.modelIR), candidateHashes: hashes, referenceHash: canonical(reference),
      scope: { surface: 'node-webgpu', runtimeHash: candidates[0].runtimeHash,
        environmentHash: canonical({ hardware: baseline.hardware, nodeVersion: process.version, providerVersion: baseline.providerVersion }),
        workloadHash: canonical(reference.cases.map(({ id, input }) => ({ id, input }))), cacheMode: 'filesystem-warm', loadMode: 'fresh-process' },
      sampling: config.sampling,
      checks: [{ id: 'tokens', mode: 'canonical-exact', maxAbsoluteError: null },
        { id: 'logits', mode: 'absolute-array', maxAbsoluteError: tolerance.logitMaxAbs },
        { id: 'scores', mode: 'absolute-array', maxAbsoluteError: tolerance.scoreMaxAbs },
        { id: 'probabilities', mode: 'absolute-array', maxAbsoluteError: tolerance.probabilityMaxAbs },
        { id: 'budget', mode: 'canonical-exact', maxAbsoluteError: null }],
      metrics: [{ id: 'peakRssBytes', unit: 'bytes', direction: 'minimize', limit: null }], selection: 'observed-range-pareto' };
    await write(`${split}-contract.json`, contract); await write(`${split}-reference.json`, reference);
    return runForgeCandidateEvaluation({ contract, reference,
      async runAttempt({ attempt, contractHash, scopeHash }) {
        assert.equal(hashBytesSha256(await fs.readFile(config.worker.path)), config.worker.digest, 'Worker changed during campaign.');
        await verifyWorkerDependencies();
        const row = rows.find(row => row.id === attempt.caseId);
        const name = `${split}-${attempt.attemptId.slice(7)}`;
        const job = { ...config.measurement, packageBundlePath: config.packageBundlePath,
          capsuleRoot: config.capsuleRoot, reference: row.sourceReference, candidate: candidateFiles.get(attempt.candidateHash),
          hardwareDigest: hashBytesSha256(Buffer.from(JSON.stringify(baseline.hardware))),
          outputDir: path.join(config.outputDir, name) };
        const jobFile = await write(`${name}.json`, job);
        const log = await fs.open(path.join(config.outputDir, `${name}.log`), 'wx');
        try {
          await new Promise((resolve, reject) => {
            const child = spawn(process.execPath, [config.worker.path, jobFile.path], { stdio: ['ignore', log.fd, log.fd] });
            const timeout = setTimeout(() => child.kill('SIGTERM'), config.timeoutMs);
            child.once('error', error => { clearTimeout(timeout); reject(error); });
            child.once('exit', (code, signal) => { clearTimeout(timeout); code === 0 ? resolve() : reject(new Error(`Measurement failed: ${code ?? signal}; see ${name}.log`)); });
          });
        } finally { await log.close(); }
        const measured = await read(path.join(job.outputDir, 'measurement.json'));
        assert(measured.physicalExecution && measured.comparison.passed && !measured.cleanupError);
        assert.equal(canonical(measured.config), canonical(job));
        assert.equal(canonical(measured.hardware), canonical(baseline.hardware));
        assert.equal(measured.selectedTargetPlanDigest, baseline.plan);
        const observed = output(measured.receipt.evidence.scores); observed.budget = measured.budgetsPassed;
        console.log(JSON.stringify({ split, case: row.id, phase: attempt.phase, run: attempt.run,
          candidate: attempt.candidateHash, peakRssBytes: measured.peakRssBytes, loadMs: measured.loadMs }));
        return { attemptId: attempt.attemptId, candidateHash: attempt.candidateHash, contractHash, scopeHash,
          inputHash: canonical(measured.receipt.evidence.query === row.reference.input.query
            ? { query: measured.receipt.evidence.query, documents: measured.receipt.evidence.documents } : null),
          status: 'completed', output: observed,
          metrics: attempt.phase === 'warmup' ? null : { peakRssBytes: measured.peakRssBytes }, error: null };
      }, onObservation: observation => write(`${split}-observation-${observation.attemptId.slice(7)}.json`, observation) });
  }
  try {
    const control = canonical(candidates.find(candidate => candidate.maxRetainedArtifactBytes === null));
    report.tuning = await evaluate('tuning', [...candidateFiles.keys()]);
    await write('tuning-evaluation.json', report.tuning);
    const selected = report.tuning.receipt.selectedCandidateHashes.filter(hash => hash !== control);
    if (selected.length !== 1 || report.tuning.receipt.selectedCandidateHashes.includes(control)) {
      report.reason = 'Tuning did not establish one candidate beyond the observed memory ranges.';
    } else {
      report.selectedCandidateHash = selected[0];
      await write('frozen-selection.json', { selectedCandidateHash: selected[0], control, tuningDigest: canonical(report.tuning) });
      report.heldout = await evaluate('heldout', [control, selected[0]]);
      await write('heldout-evaluation.json', report.heldout);
      report.correctnessPassed = [...report.tuning.receipt.candidates, ...report.heldout.receipt.candidates].every(candidate => candidate.eligible);
      report.heldoutImprovement = report.correctnessPassed && report.heldout.receipt.selectedCandidateHashes.length === 1
        && report.heldout.receipt.selectedCandidateHashes[0] === selected[0];
    }
  } finally {
    report.completedAtUtc = new Date().toISOString();
    await write('experiment.json', report);
  }
  return report;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  assert.equal(process.argv.length, 3, 'Usage: node tools/run-capsule-retention-experiment.js <config.json>');
  const report = await runRetentionExperiment(JSON.parse(await fs.readFile(process.argv[2], 'utf8')));
  console.log(JSON.stringify({ correctnessPassed: report.correctnessPassed, heldoutImprovement: report.heldoutImprovement }));
}
