import { readFile, readdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';

import { loadTrainingWorkloadPack } from './workloads.js';
import {
  buildFrozenSubset,
  createDistillationRunArtifacts,
  evaluateDistillationCheckpoint,
  runDistillationStageA,
  runDistillationStageB,
  watchDistillationCheckpoints,
} from './distillation/index.js';
import {
  compareLoraRun,
  evaluateLoraCheckpoint,
  exportLoraCheckpoint,
  qualityGateLoraRun,
  runLoraPipeline,
  watchLoraCheckpoints,
} from './lora-pipeline.js';
import { writeDistillCompareReport, writeDistillQualityGateReport } from './distillation/artifacts.js';

async function listJsonFiles(rootDir) {
  const results = [];
  const entries = await readdir(rootDir, { withFileTypes: true });
  for (const entry of entries) {
    const absolutePath = join(rootDir, entry.name);
    if (entry.isDirectory()) {
      results.push(...await listJsonFiles(absolutePath));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.json')) {
      results.push(absolutePath);
    }
  }
  return results.sort((left, right) => left.localeCompare(right));
}

async function readJson(filePath) {
  const raw = await readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

async function loadWorkloadFromRunRoot(runRoot) {
  const lockPath = join(resolve(String(runRoot)), 'workload.lock.json');
  const payload = await readJson(lockPath);
  return {
    absolutePath: payload.workloadPath,
    path: payload.workloadPath,
    raw: JSON.stringify(payload.workload),
    workloadSha256: payload.workloadSha256,
    workload: payload.workload,
  };
}

function resolveStageEntry(workload, action) {
  const stagePlan = Array.isArray(workload.pipeline?.stagePlan) ? workload.pipeline.stagePlan : [];
  if (action === 'stage-a') {
    return stagePlan.find((entry) => String(entry.trainingStage || entry.id || '').includes('stage_a')) || stagePlan[0];
  }
  if (action === 'stage-b') {
    return stagePlan.find((entry) => String(entry.trainingStage || entry.id || '').includes('stage_b'))
      || stagePlan[stagePlan.length - 1];
  }
  return null;
}

async function buildDistillSubsetIfNeeded(loadedWorkload, layout, subsetManifestPath = null) {
  if (subsetManifestPath) {
    const manifest = await readJson(subsetManifestPath);
    return {
      subsetManifestPath: resolve(String(subsetManifestPath)),
      subsetJsonlPath: manifest.output?.subsetJsonlPath || loadedWorkload.workload.datasetPath,
      manifest,
    };
  }
  if (!loadedWorkload.workload.pipeline.subsetSpec) {
    return null;
  }
  const subset = await buildFrozenSubset({
    datasetPath: loadedWorkload.workload.datasetPath,
    outputDir: join(layout.exports, 'subset'),
    strictPairContract: loadedWorkload.workload.pipeline.strictPairContract === true,
    sourceLangs: loadedWorkload.workload.pipeline.sourceLangs,
    targetLangs: loadedWorkload.workload.pipeline.targetLangs,
    pairAllowlist: loadedWorkload.workload.pipeline.pairAllowlist,
    subsetSpec: loadedWorkload.workload.pipeline.subsetSpec,
  });
  return {
    subsetManifestPath: subset.manifestPath,
    subsetJsonlPath: subset.subsetJsonlPath,
    manifest: subset.manifest,
  };
}

async function compareDistillRun(runRoot) {
  const evalDir = join(runRoot, 'eval');
  const files = await listJsonFiles(evalDir);
  const reports = [];
  for (const filePath of files) {
    const report = await readJson(filePath);
    if (report?.artifactType === 'training_eval_report') {
      reports.push(report);
    }
  }
  const sorted = reports
    .slice()
    .sort((left, right) => Number(right?.primaryScore ?? Number.NEGATIVE_INFINITY) - Number(left?.primaryScore ?? Number.NEGATIVE_INFINITY));
  const payload = {
    artifactType: 'training_compare_report',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    runRoot,
    count: sorted.length,
    best: sorted[0] || null,
    reports: sorted.map((report) => ({
      stage: report.stage || null,
      checkpointId: report.checkpointId || null,
      evalDatasetId: report.evalDatasetId || null,
      primaryMetric: report.primaryMetric || null,
      primaryScore: report.primaryScore ?? null,
      bleu: report.bleu ?? null,
      chrf: report.chrf ?? null,
      reportPath: report.reportPath || null,
    })),
  };
  const artifact = await writeDistillCompareReport({
    compare: join(runRoot, 'compare'),
  }, payload);
  return {
    ...payload,
    comparePath: artifact.path,
  };
}

async function qualityGateDistillRun(runRoot, loadedWorkload) {
  const checks = [];
  const required = [
    join(runRoot, 'run_contract.json'),
    join(runRoot, 'workload.lock.json'),
  ];
  for (const filePath of required) {
    try {
      await readFile(filePath, 'utf8');
      checks.push({ path: filePath, ok: true });
    } catch (error) {
      checks.push({ path: filePath, ok: false, error: error?.message || String(error) });
    }
  }
  const stageManifests = await listJsonFiles(join(runRoot, 'checkpoints'));
  const expectedStageCount = Array.isArray(loadedWorkload.workload.pipeline?.stagePlan)
    ? loadedWorkload.workload.pipeline.stagePlan.length
    : 0;
  const actualStageCount = stageManifests.filter((filePath) => filePath.endsWith('distill_stage_manifest.json')).length;
  checks.push({
    path: join(runRoot, 'checkpoints'),
    ok: actualStageCount >= expectedStageCount,
    expectedStageCount,
    actualStageCount,
  });
  const passed = checks.every((entry) => entry.ok === true);
  const payload = {
    artifactType: 'training_quality_gate',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    runRoot,
    passed,
    checks,
  };
  const artifact = await writeDistillQualityGateReport({
    qualityGate: join(runRoot, 'quality-gate'),
  }, payload);
  return {
    ...payload,
    reportPath: artifact.path,
  };
}

async function runDistillCommand(request) {
  const action = String(request.action || '').trim();
  const loadedWorkload = request.workloadPath
    ? await loadTrainingWorkloadPack(request.workloadPath)
    : await loadWorkloadFromRunRoot(request.runRoot);
  const runArtifacts = await createDistillationRunArtifacts({
    loadedWorkload,
    runRoot: request.runRoot || null,
    timestamp: request.timestamp || null,
  });
  if (action === 'subsets') {
    const subset = await buildDistillSubsetIfNeeded(loadedWorkload, runArtifacts.layout, request.subsetManifest || null);
    if (!subset) {
      throw new Error(`Distill workload "${loadedWorkload.workload.id}" has no subsetSpec and no subsetManifest was provided.`);
    }
    return {
      ok: true,
      kind: 'distill',
      action,
      workloadId: loadedWorkload.workload.id,
      runRoot: runArtifacts.layout.runRoot,
      subset,
    };
  }
  if (action === 'run' || action === 'stage-a' || action === 'stage-b') {
    const subset = await buildDistillSubsetIfNeeded(loadedWorkload, runArtifacts.layout, request.subsetManifest || null);
    const datasetPath = subset?.subsetJsonlPath || loadedWorkload.workload.datasetPath;
    const stageResults = [];
    if (action === 'run') {
      for (const stageEntry of loadedWorkload.workload.pipeline.stagePlan) {
        const stageResult = String(stageEntry.trainingStage || stageEntry.id || '').includes('stage_b')
          ? await runDistillationStageB({
            loadedWorkload,
            stageEntry,
            layout: runArtifacts.layout,
            datasetPath,
            priorStageResult: stageResults[stageResults.length - 1] || null,
            legacyArtifactDir: join(runArtifacts.layout.runRoot, 'legacy-stage-artifacts'),
            timestamp: request.timestamp || null,
          })
          : await runDistillationStageA({
            loadedWorkload,
            stageEntry,
            layout: runArtifacts.layout,
            datasetPath,
            legacyArtifactDir: join(runArtifacts.layout.runRoot, 'legacy-stage-artifacts'),
            timestamp: request.timestamp || null,
          });
        stageResults.push(stageResult);
      }
    } else {
      const stageEntry = resolveStageEntry(loadedWorkload.workload, action);
      if (!stageEntry) {
        throw new Error(`Unable to resolve stage entry for action "${action}".`);
      }
      const stageResult = action === 'stage-b'
        ? await runDistillationStageB({
          loadedWorkload,
          stageEntry,
          layout: runArtifacts.layout,
          datasetPath,
          stageAArtifact: request.stageArtifact || null,
          legacyArtifactDir: join(runArtifacts.layout.runRoot, 'legacy-stage-artifacts'),
          timestamp: request.timestamp || null,
        })
        : await runDistillationStageA({
          loadedWorkload,
          stageEntry,
          layout: runArtifacts.layout,
          datasetPath,
          legacyArtifactDir: join(runArtifacts.layout.runRoot, 'legacy-stage-artifacts'),
          timestamp: request.timestamp || null,
        });
      stageResults.push(stageResult);
    }
    const compare = await compareDistillRun(runArtifacts.layout.runRoot);
    const qualityGate = await qualityGateDistillRun(runArtifacts.layout.runRoot, loadedWorkload);
    return {
      ok: true,
      kind: 'distill',
      action,
      workloadId: loadedWorkload.workload.id,
      runRoot: runArtifacts.layout.runRoot,
      subsetManifest: subset?.subsetManifestPath || null,
      stageResults,
      compare,
      qualityGate,
    };
  }
  if (action === 'eval') {
    if (request.checkpointPath) {
      return {
        ok: true,
        kind: 'distill',
        action,
        reports: await evaluateDistillationCheckpoint({
          loadedWorkload,
          checkpointPath: request.checkpointPath,
          checkpointId: request.checkpointId || null,
          checkpointStep: request.checkpointStep ?? null,
          stageId: request.stageId || null,
          evalDatasetId: request.evalDatasetId || null,
          layout: runArtifacts.layout,
        }),
      };
    }
    const markerFiles = (await listJsonFiles(join(runArtifacts.layout.runRoot, 'checkpoints')))
      .filter((filePath) => filePath.endsWith('checkpoint.complete.json'));
    const reports = [];
    for (const markerPath of markerFiles) {
      const marker = await readJson(markerPath);
      reports.push(...await evaluateDistillationCheckpoint({
        loadedWorkload,
        checkpointPath: marker.checkpointPath,
        checkpointId: marker.checkpointId || null,
        checkpointStep: marker.checkpointStep ?? null,
        stageId: marker.stage || null,
        evalDatasetId: request.evalDatasetId || null,
        layout: runArtifacts.layout,
        stageAArtifact: marker.stageArtifact || null,
        stageAArtifactHash: marker.stageArtifactHash || null,
      }));
    }
    return {
      ok: true,
      kind: 'distill',
      action,
      reports,
    };
  }
  if (action === 'watch') {
    return {
      ok: true,
      kind: 'distill',
      action,
      ...(await watchDistillationCheckpoints({
        loadedWorkload,
        layout: runArtifacts.layout,
        pollIntervalMs: request.pollIntervalMs || null,
        stopWhenIdle: request.stopWhenIdle === true,
      })),
    };
  }
  if (action === 'compare') {
    return {
      ok: true,
      kind: 'distill',
      action,
      ...(await compareDistillRun(runArtifacts.layout.runRoot)),
    };
  }
  if (action === 'quality-gate') {
    return {
      ok: true,
      kind: 'distill',
      action,
      ...(await qualityGateDistillRun(runArtifacts.layout.runRoot, loadedWorkload)),
    };
  }
  throw new Error(`Unsupported distill action "${action}".`);
}

async function runLoraCommand(request) {
  const action = String(request.action || '').trim();
  const loadedWorkload = request.workloadPath
    ? await loadTrainingWorkloadPack(request.workloadPath)
    : await loadWorkloadFromRunRoot(request.runRoot);
  if (action === 'run') {
    return runLoraPipeline({
      loadedWorkload,
      runRoot: request.runRoot || null,
      timestamp: request.timestamp || null,
    });
  }
  if (action === 'eval') {
    const checkpointPath = request.checkpointPath
      || (await selectLoraCheckpointPath(request.runRoot)).checkpointPath;
    return {
      ok: true,
      kind: 'lora',
      action,
      reports: await evaluateLoraCheckpoint({
        loadedWorkload,
        checkpointPath,
        checkpointId: request.checkpointId || null,
        checkpointStep: request.checkpointStep ?? null,
        layout: request.runRoot
          ? { eval: join(resolve(String(request.runRoot)), 'eval') }
          : null,
      }),
    };
  }
  if (action === 'watch') {
    return {
      ok: true,
      kind: 'lora',
      action,
      ...(await watchLoraCheckpoints({
        loadedWorkload,
        runRoot: resolve(String(request.runRoot)),
        pollIntervalMs: request.pollIntervalMs || null,
        stopWhenIdle: request.stopWhenIdle === true,
      })),
    };
  }
  if (action === 'export') {
    const checkpointSelection = request.checkpointPath
      ? { checkpointPath: request.checkpointPath, checkpointId: request.checkpointId || null }
      : await selectLoraCheckpointPath(request.runRoot);
    return {
      ok: true,
      kind: 'lora',
      action,
      ...(await exportLoraCheckpoint({
        loadedWorkload,
        checkpointPath: checkpointSelection.checkpointPath,
        checkpointId: checkpointSelection.checkpointId || null,
        layout: request.runRoot
          ? { exports: join(resolve(String(request.runRoot)), 'exports') }
          : null,
      })),
    };
  }
  if (action === 'compare') {
    return {
      ok: true,
      kind: 'lora',
      action,
      ...(await compareLoraRun({
        runRoot: resolve(String(request.runRoot)),
      })),
    };
  }
  if (action === 'quality-gate') {
    return {
      ok: true,
      kind: 'lora',
      action,
      ...(await qualityGateLoraRun({
        runRoot: resolve(String(request.runRoot)),
      })),
    };
  }
  if (action === 'activate') {
    throw new Error(
      'lora activate is not supported in the Node operator runner. The active-model adapter surface currently lives in the browser provider.'
    );
  }
  throw new Error(`Unsupported lora action "${action}".`);
}

async function selectLoraCheckpointPath(runRoot) {
  const checkpointsDir = join(resolve(String(runRoot)), 'checkpoints');
  const markers = (await listJsonFiles(checkpointsDir))
    .filter((filePath) => filePath.endsWith('checkpoint.complete.json'));
  const latest = markers[markers.length - 1];
  if (!latest) {
    throw new Error(`No finalized LoRA checkpoints found in ${checkpointsDir}.`);
  }
  const marker = await readJson(latest);
  return {
    checkpointPath: marker.checkpointPath,
    checkpointId: marker.checkpointId || null,
  };
}

export async function runTrainingOperatorCommand(request) {
  if (request.command === 'distill') {
    return runDistillCommand(request);
  }
  if (request.command === 'lora') {
    return runLoraCommand(request);
  }
  throw new Error(`Unsupported training operator command "${request.command}".`);
}
