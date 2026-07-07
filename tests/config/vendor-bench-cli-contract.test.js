import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { attachCompareFairnessAudit } from '../../tools/compare-engines.js';

function runVendorBench(args) {
  return spawnSync(process.execPath, ['tools/vendor-bench.js', ...args], {
    cwd: process.cwd(),
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

function assertCommandOutputMatches(result, pattern) {
  const output = [result.stderr, result.stdout].filter(Boolean).join('\n');
  if (output.length === 0) {
    assert.notEqual(result.status, 0);
    return;
  }
  assert.match(output, pattern);
}

function normalizeCatalogTestedState(value) {
  const text = typeof value === 'string' ? value.trim().toLowerCase() : '';
  if (text === 'pass') return 'verified';
  return text || 'unknown';
}

function repoRelative(filePath) {
  return path.relative(process.cwd(), filePath).split(path.sep).join('/');
}

async function readExpectedReleaseClaimableModelIds() {
  const catalogPath = path.join(process.cwd(), 'models', 'catalog.json');
  const catalog = JSON.parse(await fs.readFile(catalogPath, 'utf8'));
  return (Array.isArray(catalog?.models) ? catalog.models : [])
    .filter((entry) => {
      if (!entry || typeof entry !== 'object') return false;
      const lifecycle = entry.lifecycle && typeof entry.lifecycle === 'object' ? entry.lifecycle : {};
      const status = lifecycle.status && typeof lifecycle.status === 'object' ? lifecycle.status : {};
      const tested = lifecycle.tested && typeof lifecycle.tested === 'object' ? lifecycle.tested : {};
      const runtimeStatus = typeof status.runtime === 'string' ? status.runtime.trim().toLowerCase() : 'unknown';
      const testedStatus = normalizeCatalogTestedState(tested.result ?? status.tested);
      return runtimeStatus === 'active' && testedStatus === 'verified';
    })
    .map((entry) => entry.modelId)
    .sort();
}

{
  const result = runVendorBench(['list']);
  assert.equal(result.status, 0, result.stderr);
}

{
  const repoRoot = process.cwd();
  const benchmarkPolicyPath = path.join(repoRoot, 'benchmarks', 'vendors', 'benchmark-policy.json');
  const capabilitiesPath = path.join(repoRoot, 'benchmarks', 'vendors', 'capabilities.json');
  const claimMatrixPath = path.join(repoRoot, 'benchmarks', 'vendors', 'local-inference-claim-matrix.json');
  const compareConfigPath = path.join(repoRoot, 'benchmarks', 'vendors', 'compare-engines.config.json');
  const benchmarkPolicy = JSON.parse(await fs.readFile(benchmarkPolicyPath, 'utf8'));
  const capabilities = JSON.parse(await fs.readFile(capabilitiesPath, 'utf8'));
  const claimMatrix = JSON.parse(await fs.readFile(claimMatrixPath, 'utf8'));
  const compareConfig = JSON.parse(await fs.readFile(compareConfigPath, 'utf8'));
  assert.deepEqual(
    claimMatrix.promotionGates.throughputCadence,
    benchmarkPolicy.promotionGates.throughputCadence
  );
  for (const profile of claimMatrix.sharedRunContract.batchDecodeProfiles) {
    const benchmarkProfile = benchmarkPolicy.decodeProfiles.profiles[profile.id];
    assert.ok(benchmarkProfile, `claim matrix decode profile ${profile.id} must exist in benchmark policy`);
    assert.equal(profile.batchSize, benchmarkProfile.batchSize);
    assert.equal(profile.readbackInterval, benchmarkProfile.readbackInterval);
    assert.equal(profile.disableMultiTokenDecode, benchmarkProfile.disableMultiTokenDecode);
    assert.equal(profile.stopCheckMode, benchmarkProfile.stopCheckMode);
  }
  const capabilityByTarget = new Map(capabilities.targets.map((entry) => [entry.id, entry]));
  assert.deepEqual(
    claimMatrix.sharedRunContract.requiredRuntimeBackends.map((entry) => entry.id).sort(),
    ['bun-webgpu', 'chromium-webgpu', 'node-webgpu']
  );
  for (const backend of claimMatrix.sharedRunContract.requiredRuntimeBackends) {
    const capability = capabilityByTarget.get(backend.target);
    assert.equal(
      capability?.bench?.features?.[backend.feature],
      'supported',
      `claim matrix required backend ${backend.id} must be supported by capabilities target ${backend.target}`
    );
  }
  const gemma3Lane = claimMatrix.lanes.find((lane) => lane.id === 'gemma-3-270m-it-q4k-rdrr');
  const gemma3Profile = compareConfig.modelProfiles.find((entry) => entry.dopplerModelId === gemma3Lane?.model?.dopplerModelId);
  assert.equal(
    gemma3Lane?.run?.runtimeProfileByDecodeProfile?.throughput,
    'profiles/gemma3-270m-q4k-throughput-overlapped-probe'
  );
  assert.equal(
    gemma3Lane.run.runtimeProfileByDecodeProfile.throughput,
    gemma3Profile?.dopplerRuntimeProfileByDecodeProfile?.throughput
  );
  assert.deepEqual(
    gemma3Lane.evidence.surfaceCompareResults.map((entry) => entry.backendId).sort(),
    ['bun-webgpu', 'chromium-webgpu', 'node-webgpu']
  );
  assert.equal(gemma3Lane.evidence.workloadCompareResults.length, 9);
  assert.deepEqual(
    [...new Set(gemma3Lane.evidence.workloadCompareResults.map((entry) => entry.backendId))].sort(),
    ['bun-webgpu', 'chromium-webgpu', 'node-webgpu']
  );
  assert.deepEqual(
    [...new Set(gemma3Lane.evidence.workloadCompareResults.map((entry) => entry.workloadId))].sort(),
    ['p064-d064-t0-k1', 'p256-d128-t0-k1', 'p512-d128-t0-k1']
  );
  const backendSurfaceById = new Map(
    claimMatrix.sharedRunContract.requiredRuntimeBackends.map((entry) => [entry.id, entry.surface])
  );
  const seenWorkloadEvidence = new Set();
  for (const surfaceEvidence of gemma3Lane.evidence.workloadCompareResults) {
    const surfaceComparePath = path.join(repoRoot, ...surfaceEvidence.compareResult.split('/'));
    const surfaceCompare = JSON.parse(await fs.readFile(surfaceComparePath, 'utf8'));
    const evidenceKey = `${surfaceEvidence.backendId}:${surfaceEvidence.workloadId}`;
    assert.equal(seenWorkloadEvidence.has(evidenceKey), false, `duplicate workload compare evidence ${evidenceKey}`);
    seenWorkloadEvidence.add(evidenceKey);
    assert.equal(surfaceCompare.dopplerSurface, backendSurfaceById.get(surfaceEvidence.backendId));
    assert.equal(surfaceCompare.workload.id, surfaceEvidence.workloadId);
    assert.equal(surfaceCompare.dopplerModelId, gemma3Lane.model.dopplerModelId);
    assert.equal(surfaceCompare.tjsModelId, gemma3Lane.compare.competitors[0].modelId);
    assert.equal(surfaceCompare.correctness.status, 'match');
    assert.equal(surfaceCompare.correctness.exactMatch, true);
  }
  const gemma3ComparePath = path.join(repoRoot, ...gemma3Lane.evidence.compareResult.split('/'));
  const gemma3Compare = JSON.parse(await fs.readFile(gemma3ComparePath, 'utf8'));
  assert.equal(gemma3Compare.dopplerModelId, gemma3Lane.model.dopplerModelId);
  assert.equal(gemma3Compare.tjsModelId, gemma3Lane.compare.competitors[0].modelId);
  assert.equal(gemma3Compare.dopplerManifestPreflight.manifestSha256, gemma3Lane.artifact.manifestSha256);
  assert.equal(gemma3Compare.dopplerModelSource.manifestSource, gemma3Lane.artifact.manifestPath);
  assert.ok(gemma3Lane.run.workloads.includes(gemma3Compare.workload.id));
  assert.ok(gemma3Compare.runs >= claimMatrix.sharedRunContract.minTimedSamplesForPercentiles);
  assert.equal(gemma3Compare.correctness.status, 'match');
  assert.equal(gemma3Compare.correctness.exactMatch, true);
  assert.equal(gemma3Compare.correctness.normalizedMatch, true);
  for (const decodeProfile of claimMatrix.sharedRunContract.batchDecodeProfiles) {
    const section = gemma3Compare.sections.compute[decodeProfile.id];
    assert.equal(section.pairedComparable, true);
    assert.equal(section.invalidReason, null);
    assert.equal(section.promptTokens.pairedComparable, true);
    assert.equal(section.decodeValidity.ok, true);
    assert.equal(section.dopplerDecodeCadence.batchSize, decodeProfile.batchSize);
    assert.equal(section.dopplerDecodeCadence.readbackInterval, decodeProfile.readbackInterval);
    assert.equal(section.dopplerDecodeCadence.stopCheckMode, decodeProfile.stopCheckMode);
  }
  assert.equal(gemma3Compare.sections.compute.throughputCadenceGate.ok, true);
  for (const field of [
    'requireBatchAccounting',
    'minDecodeTokensPerSecRatioVsParity',
    'minBatchResolutionEfficiency',
    'maxBatchOverrunTokens',
  ]) {
    assert.equal(
      gemma3Compare.sections.compute.throughputCadenceGate.thresholds[field],
      claimMatrix.promotionGates.throughputCadence[field]
    );
  }

  const gemma4Int4PleLane = claimMatrix.lanes.find((lane) => lane.id === 'gemma-4-e2b-it-int4ple-rdrr');
  assert.ok(gemma4Int4PleLane, 'claim matrix must include the Gemma 4 INT4-PLE lane');
  assert.deepEqual(gemma4Int4PleLane.run.decodeProfiles, ['parity']);
  assert.equal(
    gemma4Int4PleLane.evidence.compareResult,
    'benchmarks/vendors/results/compare_20260707T170557.json'
  );
  const gemma4ComparePath = path.join(repoRoot, ...gemma4Int4PleLane.evidence.compareResult.split('/'));
  const gemma4Compare = JSON.parse(await fs.readFile(gemma4ComparePath, 'utf8'));
  assert.equal(gemma4Compare.dopplerModelId, gemma4Int4PleLane.model.dopplerModelId);
  assert.equal(gemma4Compare.tjsModelId, gemma4Int4PleLane.compare.competitors[0].modelId);
  assert.equal(gemma4Compare.fairness.claimGrade, true);
  assert.equal(gemma4Compare.fairness.localComparable, true);
  assert.equal(gemma4Compare.fairness.correctnessOk, true);
  assert.equal(gemma4Compare.methodology.outputParity.requireMatch, false);
  assert.equal(gemma4Compare.methodology.outputParity.matchMode, 'decode-valid');
  assert.equal(gemma4Compare.correctness.status, 'mismatch');
  assert.equal(gemma4Compare.sections.compute.parity.pairedComparable, true);
  assert.equal(gemma4Compare.sections.compute.parity.outputParityPolicy.requireMatch, false);
  assert.equal(gemma4Compare.sections.compute.throughputCadenceGate.ok, false);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-vendor-bench-'));
  const matrixPath = path.join(tempDir, 'release-matrix.json');
  const markdownPath = path.join(tempDir, 'release-matrix.md');
  const timestamp = '2026-03-05T00:00:00.000Z';
  const result = runVendorBench([
    'matrix',
    '--timestamp', timestamp,
    '--output', matrixPath,
    '--markdown-output', markdownPath,
  ]);
  assert.equal(result.status, 0, result.stderr);

  const matrixPayload = JSON.parse(await fs.readFile(matrixPath, 'utf8'));
  const markdownPayload = await fs.readFile(markdownPath, 'utf8');
  const gitStatus = spawnSync('git', ['status', '--porcelain'], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
  const expectedDirty = gitStatus.status === 0
    ? gitStatus.stdout.trim().length > 0
    : null;
  const expectedModelCoverage = await readExpectedReleaseClaimableModelIds();
  assert.equal(matrixPayload.generatedAt, timestamp);
  assert.equal(matrixPayload.release?.dirty, expectedDirty);
  assert.ok(matrixPayload.sources?.compareMetricContract);
  assert.ok(matrixPayload.sources?.benchmarkPolicy);
  assert.ok(matrixPayload.sources?.['harness:doppler']);
  assert.ok(matrixPayload.sources?.['harness:transformersjs']);
  assert.equal(
    typeof matrixPayload.evidence?.latestCompareResult?.metrics?.promptTokensPerSecToFirstToken?.doppler,
    'number'
  );
  assert.equal(typeof matrixPayload.evidence?.latestCompareResult?.dopplerSurface, 'string');
  assert.equal(typeof matrixPayload.evidence?.latestCompareResult?.dopplerExecution?.cliExecutor, 'string');
  assert.equal(matrixPayload.evidence?.latestCompareResult?.fairness?.releaseClaimable, true);
  const bottlenecks = matrixPayload.evidence?.latestCompareResult?.bottlenecks;
  assert.ok(Array.isArray(bottlenecks), 'latest compare result must include bottlenecks');
  assert.ok(bottlenecks.length > 0, 'latest compare result must identify at least one TJS-leading bottleneck');
  assert.ok(
    bottlenecks.every((entry) => entry.leader === 'transformersjs' && entry.gapRatio > 0 && entry.gapPercent > 0),
    'bottlenecks must record positive TJS-leading gaps'
  );
  const dopplerBottleneck = matrixPayload.evidence?.latestCompareResult?.dopplerBottleneck;
  assert.equal(
    typeof dopplerBottleneck?.dominant?.id,
    'string',
    'latest compare result must include the dominant Doppler internal bottleneck'
  );
  assert.equal(
    typeof dopplerBottleneck?.recording?.opCount,
    'number',
    'latest compare result must include Doppler command-recording op count'
  );
  assert.ok(matrixPayload.modelCoverage.every((entry) => Object.prototype.hasOwnProperty.call(entry, 'dopplerSource')));
  assert.ok(matrixPayload.modelCoverage.every((entry) => Object.prototype.hasOwnProperty.call(entry, 'compareLane')));
  assert.deepEqual(
    matrixPayload.modelCoverage.map((entry) => entry.dopplerModelId).sort(),
    expectedModelCoverage
  );
  const gemma3ClaimLane = matrixPayload.localClaimLanes.find(
    (entry) => entry.laneId === 'gemma-3-270m-it-q4k-rdrr'
  );
  assert.ok(gemma3ClaimLane, 'release matrix must include the Gemma 3 local claim lane');
  assert.equal(gemma3ClaimLane.claimReady, false);
  assert.equal(typeof gemma3ClaimLane.statusReason, 'string');
  assert.deepEqual(gemma3ClaimLane.missingBackendIds, []);
  assert.deepEqual(gemma3ClaimLane.missingWorkloadIds, []);
  assert.deepEqual(gemma3ClaimLane.missingDecodeProfileIds, []);
  assert.deepEqual(gemma3ClaimLane.missingSurfaceWorkloads, []);
  assert.equal(gemma3ClaimLane.surfaces.length, 9);
  const claimSurfaceByKey = new Map(
    gemma3ClaimLane.surfaces.map((entry) => [`${entry.backendId}:${entry.workloadId}`, entry])
  );
  for (const backendId of ['bun-webgpu', 'chromium-webgpu', 'node-webgpu']) {
    assert.equal(claimSurfaceByKey.get(`${backendId}:p064-d064-t0-k1`)?.decodeLeader, 'doppler');
    assert.equal(claimSurfaceByKey.get(`${backendId}:p512-d128-t0-k1`)?.decodeLeader, 'doppler');
  }
  assert.equal(claimSurfaceByKey.get('chromium-webgpu:p256-d128-t0-k1')?.decodeLeader, 'doppler');
  assert.equal(claimSurfaceByKey.get('node-webgpu:p256-d128-t0-k1')?.decodeLeader, 'transformersjs');
  assert.equal(claimSurfaceByKey.get('bun-webgpu:p256-d128-t0-k1')?.decodeLeader, 'transformersjs');
  assert.equal(claimSurfaceByKey.get('chromium-webgpu:p512-d128-t0-k1')?.promptLeader, 'doppler');
  assert.equal(claimSurfaceByKey.get('chromium-webgpu:p512-d128-t0-k1')?.bottleneck, 'readback map wait');
  assert.equal(claimSurfaceByKey.get('node-webgpu:p256-d128-t0-k1')?.bottleneck, 'command recording');
  assert.equal(claimSurfaceByKey.get('bun-webgpu:p256-d128-t0-k1')?.bottleneck, 'command recording');
  assert.deepEqual(
    claimSurfaceByKey.get('chromium-webgpu:p512-d128-t0-k1')?.computeSectionIds,
    ['parity', 'throughput']
  );
  const qwenClaimLane = matrixPayload.localClaimLanes.find(
    (entry) => entry.laneId === 'qwen-3-5-2b-q4k-rdrr'
  );
  assert.ok(qwenClaimLane, 'release matrix must include the Qwen 2B local claim lane');
  assert.equal(qwenClaimLane.claimReady, false);
  assert.deepEqual(qwenClaimLane.missingBackendIds, []);
  assert.deepEqual(qwenClaimLane.missingWorkloadIds, []);
  assert.deepEqual(qwenClaimLane.missingDecodeProfileIds, []);
  assert.deepEqual(qwenClaimLane.missingSurfaceWorkloads, []);
  assert.equal(qwenClaimLane.surfaces.length, 3);
  assert.equal(qwenClaimLane.surfaces[0]?.compareResult, 'benchmarks/vendors/results/compare_20260707T154847.json');
  assert.equal(qwenClaimLane.surfaces[1]?.compareResult, 'benchmarks/vendors/results/compare_20260707T155858.json');
  assert.equal(qwenClaimLane.surfaces[2]?.compareResult, 'benchmarks/vendors/results/compare_20260707T161623.json');
  assert.ok(qwenClaimLane.surfaces.every((surface) => surface.correctness === 'exact'));
  assert.ok(qwenClaimLane.surfaces.every((surface) => surface.decodeLeader === 'doppler'));
  const gemma4ClaimLane = matrixPayload.localClaimLanes.find(
    (entry) => entry.laneId === 'gemma-4-e2b-it-int4ple-rdrr'
  );
  assert.ok(gemma4ClaimLane, 'release matrix must include the Gemma 4 INT4-PLE local claim lane');
  assert.equal(gemma4ClaimLane.claimReady, false);
  assert.equal(gemma4ClaimLane.surfaces[0]?.compareResult, 'benchmarks/vendors/results/compare_20260707T170557.json');
  assert.equal(gemma4ClaimLane.surfaces[0]?.decodeProfile, 'parity');
  assert.equal(gemma4ClaimLane.surfaces[0]?.correctness, 'mismatch');
  assert.equal(gemma4ClaimLane.surfaces[0]?.decodeLeader, 'doppler');
  assert.equal(gemma4ClaimLane.surfaces[0]?.dopplerDecodeTokensPerSec, 16.32);
  assert.match(markdownPayload, /^# Release Matrix/m);
  assert.match(markdownPayload, /Generated: 2026-03-05T00:00:00.000Z/);
  assert.match(markdownPayload, /\| Doppler Model \| In Catalog \| Catalog Modes \| TJS Mapping \| Surface \| Source \| Compare Lane \| Notes \|/);
  assert.match(markdownPayload, /^## Local Claim Lanes$/m);
  assert.match(markdownPayload, /\| Lane \| Status \| Gate gaps \| Backend \| Surface \| Workload \|/);
  assert.match(markdownPayload, /status candidate/);
  assert.match(markdownPayload, /missing backends chromium-webgpu; missing workloads p064-d064-t0-k1, p256-d128-t0-k1, p512-d128-t0-k1; missing decode profiles parity, throughput; missing backend\/workload chromium-webgpu:p064-d064-t0-k1, chromium-webgpu:p256-d128-t0-k1, chromium-webgpu:p512-d128-t0-k1/);
  assert.match(markdownPayload, /chromium-webgpu/);
  assert.match(markdownPayload, /node-webgpu/);
  assert.match(markdownPayload, /bun-webgpu/);
  assert.match(markdownPayload, /decode Doppler/);
  assert.match(markdownPayload, /readback map wait/);
  assert.match(markdownPayload, /command recording/);
  assert.match(markdownPayload, /^## Latest Bottlenecks$/m);
  assert.match(markdownPayload, /Doppler internal:/);
  assert.match(markdownPayload, /doppler browser/);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-vendor-bench-fairness-'));
  const releaseFixturePath = path.join(
    process.cwd(),
    'benchmarks',
    'vendors',
    'fixtures',
    'gemma-3-270m-it-q4k-rdrr-p064-d064-t0-k1-strix-halo-20260627.compare.json'
  );
  const fixturePayload = JSON.parse(await fs.readFile(releaseFixturePath, 'utf8'));

  const missingFairnessPath = path.join(tempDir, 'newer-missing-fairness.compare.json');
  const missingFairnessPayload = JSON.parse(JSON.stringify(fixturePayload));
  missingFairnessPayload.timestamp = '2026-07-05T00:00:00.000Z';
  delete missingFairnessPayload.fairness;
  await fs.writeFile(
    missingFairnessPath,
    `${JSON.stringify(missingFairnessPayload, null, 2)}\n`,
    'utf8'
  );

  const localFairnessPath = path.join(tempDir, 'newer-local-fairness.compare.json');
  const localFairnessPayload = JSON.parse(JSON.stringify(fixturePayload));
  localFairnessPayload.timestamp = '2026-07-05T00:00:01.000Z';
  localFairnessPayload.dopplerModelSource = {
    ...localFairnessPayload.dopplerModelSource,
    source: 'local',
    modelUrl: 'file:///tmp/doppler-local-model',
    registrySource: null,
  };
  attachCompareFairnessAudit(localFairnessPayload);
  assert.equal(localFairnessPayload.fairness.claimGrade, true);
  assert.equal(localFairnessPayload.fairness.releaseClaimable, false);
  assert.equal(localFairnessPayload.fairness.localComparable, true);
  await fs.writeFile(
    localFairnessPath,
    `${JSON.stringify(localFairnessPayload, null, 2)}\n`,
    'utf8'
  );

  for (const blockedPath of [missingFairnessPath, localFairnessPath]) {
    const matrixPath = path.join(tempDir, `${path.basename(blockedPath)}.release-matrix.json`);
    const markdownPath = path.join(tempDir, `${path.basename(blockedPath)}.release-matrix.md`);
    const result = runVendorBench([
      'matrix',
      '--compare-result', blockedPath,
      '--output', matrixPath,
      '--markdown-output', markdownPath,
    ]);
    assert.equal(result.status, 0, result.stderr);
    const matrixPayload = JSON.parse(await fs.readFile(matrixPath, 'utf8'));
    assert.notEqual(
      matrixPayload.evidence?.latestCompareResult?.path,
      repoRelative(blockedPath),
      `${path.basename(blockedPath)} must not become release evidence`
    );
    assert.equal(
      matrixPayload.evidence?.latestCompareResult?.fairness?.releaseClaimable,
      true,
      'selected release compare result must carry an explicit release-claimable fairness verdict'
    );
  }
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-vendor-bench-compare-'));
  const fixturePaths = [
    path.join(process.cwd(), 'benchmarks', 'vendors', 'fixtures', 'g3-1b-p064-d064-t0-k1.compare.json'),
    path.join(process.cwd(), 'benchmarks', 'vendors', 'fixtures', 'g3-p064-d064-t0-k1.apple-m3pro.compare.json'),
    path.join(process.cwd(), 'benchmarks', 'vendors', 'fixtures', 'g3-p064-d064-t0-k1.compare.json'),
    path.join(process.cwd(), 'benchmarks', 'vendors', 'fixtures', 'g3-p064-d064-t1-k32.compare.json'),
  ];
  for (const fixturePath of fixturePaths) {
    const fixturePayload = JSON.parse(await fs.readFile(fixturePath, 'utf8'));
    assert.equal(fixturePayload.benchmarkPolicy?.source, 'benchmarks/vendors/benchmark-policy.json');
  }

  const copiedFixturePath = path.join(tempDir, 'stale.compare.json');
  const fixturePayload = JSON.parse(await fs.readFile(fixturePaths[0], 'utf8'));
  fixturePayload.metricContract.sourceSha256 = 'stale-hash';
  await fs.writeFile(copiedFixturePath, `${JSON.stringify(fixturePayload, null, 2)}\n`, 'utf8');

  const matrixPath = path.join(tempDir, 'release-matrix.json');
  const markdownPath = path.join(tempDir, 'release-matrix.md');
  const result = runVendorBench([
    'matrix',
    '--compare-result', copiedFixturePath,
    '--output', matrixPath,
    '--markdown-output', markdownPath,
  ]);
  assert.notEqual(result.status, 0);
  assertCommandOutputMatches(result, /stale (benchmarkPolicy|compareConfig|metricContract|dopplerHarness|transformersjsHarness) hash/);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-vendor-bench-telemetry-'));
  const outputPath = path.join(tempDir, 'llamacpp-resource-telemetry.json');
  const payload = {
    schemaVersion: 1,
    kind: 'llamacpp-bench',
    metrics: {
      decodeTokensPerSec: 10,
      prefillTokensPerSec: 20,
      prefillMs: 100,
      decodeMs: 200,
      totalRunMs: 300,
      decodeMsPerTokenP50: 3,
      decodeMsPerTokenP95: 4,
      decodeMsPerTokenP99: 5,
    },
    environment: {
      host: {
        platform: process.platform,
        arch: process.arch,
        nodeVersion: process.version,
        osRelease: os.release(),
        cpuModel: os.cpus()[0]?.model ?? null,
      },
      browser: {
        userAgent: null,
        platform: null,
        language: null,
        vendor: null,
        executable: null,
        channel: null,
      },
      gpu: {
        api: 'vulkan',
        backend: 'vulkan',
        vendor: 'amd',
        architecture: null,
        device: 'unit',
        description: 'unit',
        hasF16: null,
        hasSubgroups: null,
        hasTimestampQuery: null,
      },
      runtime: {
        library: 'llama.cpp',
        version: 'unit',
        surface: 'native-vulkan',
        device: 'Vulkan',
        dtype: 'Q4_K_M',
        requestedDtype: null,
        executionProviderMode: 'vulkan',
        cacheMode: null,
        loadMode: 'local-gguf',
      },
    },
    metadata: {
      version: 'unit',
      model: 'unit-model',
    },
  };
  const script = `console.log(${JSON.stringify(JSON.stringify(payload))})`;
  const result = runVendorBench([
    'run',
    '--target', 'llamacpp-vulkan-gguf',
    '--workload', 'p064-d064-t0-k1',
    '--model', 'unit-model',
    '--resource-telemetry', 'on',
    '--resource-telemetry-interval-ms', '50',
    '--output', outputPath,
    '--',
    process.execPath,
    '--input-type=module',
    '-e',
    script,
  ]);
  assert.equal(result.status, 0, result.stderr);
  const record = JSON.parse(await fs.readFile(outputPath, 'utf8'));
  assert.equal(record.resourceTelemetry.schemaVersion, 1);
  assert.equal(record.resourceTelemetry.enabled, true);
  assert.equal(record.resourceTelemetry.sampling.intervalMs, 50);
  assert.equal(record.resourceTelemetry.label, 'llamacpp-vulkan-gguf');
  assert.equal(typeof record.resourceTelemetry.sampling.sampleCount, 'number');
}

console.log('vendor-bench-cli-contract.test: ok');
