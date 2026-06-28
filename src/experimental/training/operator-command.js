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
import { summarizeAgentEvalReportRequirements } from './operator-agent-eval.js';
import {
  buildDistillArtifactBase,
  writeDistillCompareReport,
  writeDistillQualityGateReport,
  writeDistillStageManifest,
} from './distillation/artifacts.js';
import { sha256Hex } from '../../utils/sha256.js';

async function listJsonFiles(rootDir) {
  const results = [];
  let entries = [];
  try {
    entries = await readdir(rootDir, { withFileTypes: true });
  } catch (error) {
    if (error?.code === 'ENOENT') return results;
    throw error;
  }
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

function resolveComparableReportMetric(report, metric) {
  if (!report || typeof report !== 'object') return null;
  const direct = report[metric];
  if (typeof direct === 'number' && Number.isFinite(direct)) {
    return direct;
  }
  const nested = report.metrics?.[metric];
  if (typeof nested === 'number' && Number.isFinite(nested)) {
    return nested;
  }
  if (nested && typeof nested === 'object' && typeof nested.score === 'number' && Number.isFinite(nested.score)) {
    return nested.score;
  }
  return null;
}

function sortTrainingReports(reports, workload = null) {
  const metric = String(workload?.selectionMetric || reports[0]?.primaryMetric || 'primaryScore').trim();
  const goal = String(workload?.selectionGoal || (metric === 'loss' ? 'min' : 'max')).trim();
  return reports
    .slice()
    .sort((left, right) => {
      const missingScore = goal === 'min' ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
      const leftScore = resolveComparableReportMetric(left, metric) ?? missingScore;
      const rightScore = resolveComparableReportMetric(right, metric) ?? missingScore;
      if (goal === 'min') return leftScore - rightScore;
      return rightScore - leftScore;
    });
}

function collectRequiredImprovementEvalIds(workload) {
  const evalDatasets = Array.isArray(workload?.evalDatasets) ? workload.evalDatasets : [];
  return evalDatasets
    .filter((entry) => entry?.quality?.requireImprovement === true)
    .map((entry) => entry.id);
}

function summarizeQualityClaims(reports) {
  const claims = reports
    .map((report) => report.qualityClaim)
    .filter((claim) => claim && typeof claim === 'object');
  return {
    count: claims.length,
    improvedCount: claims.filter((claim) => claim.improved === true).length,
    requiredCount: claims.filter((claim) => claim.requireImprovement === true).length,
    failedRequiredCount: claims.filter((claim) => claim.requireImprovement === true && claim.improved !== true).length,
  };
}

async function loadTrainingEvalReports(rootDir) {
  const files = await listJsonFiles(rootDir);
  const reports = [];
  for (const filePath of files) {
    const report = await readJson(filePath);
    if (report?.artifactType === 'training_eval_report') {
      reports.push({
        ...report,
        reportPath: report.reportPath || filePath,
      });
    }
  }
  return reports;
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

function normalizeStageLabel(value) {
  return String(value || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
}

function isSftDistillStage(stageEntry) {
  const trainingStage = normalizeStageLabel(stageEntry?.trainingStage || stageEntry?.id || '');
  const objective = normalizeStageLabel(stageEntry?.objective || '');
  return trainingStage === 'sft' || objective === 'sft';
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function buildSftLoraLoadedWorkload(loadedWorkload, stageEntry, datasetPath) {
  const workload = loadedWorkload.workload;
  const sftLora = workload.pipeline?.sftLora || null;
  if (!sftLora) {
    throw new Error(`Distill SFT stage "${stageEntry.id}" requires distill.sftLora.`);
  }
  const stageId = String(stageEntry.id || 'sft').trim();
  const syntheticWorkload = cloneJson({
    ...workload,
    kind: 'lora',
    id: `${workload.id}-${stageId}-lora`,
    description: `${workload.description} SFT LoRA stage ${stageId}`,
    baseModelId: workload.studentModelId || workload.baseModelId,
    datasetPath,
    checkpointEvery: stageEntry.checkpointEvery,
    selectionMetric: stageEntry.selectionMetric,
    selectionGoal: stageEntry.selectionGoal,
    training: {
      ...workload.training,
      steps: stageEntry.steps,
    },
    pipeline: sftLora,
  });
  const raw = JSON.stringify(syntheticWorkload);
  return {
    ...loadedWorkload,
    raw,
    workloadSha256: sha256Hex(raw),
    workload: syntheticWorkload,
  };
}

function resolveLoraDatasetHash(loraResult) {
  return loraResult?.dataset?.datasetHash
    || loraResult?.exports?.[0]?.manifest?.metadata?.datasetHash
    || null;
}

async function runDistillSftLoraStage(options) {
  const loadedWorkload = options.loadedWorkload;
  const stageEntry = options.stageEntry;
  const stageId = String(stageEntry.id || 'sft').trim();
  const loraLoadedWorkload = buildSftLoraLoadedWorkload(
    loadedWorkload,
    stageEntry,
    options.datasetPath || loadedWorkload.workload.datasetPath
  );
  const loraResult = await runLoraPipeline({
    loadedWorkload: loraLoadedWorkload,
    runRoot: join(options.layout.checkpoints, stageId, 'sft-lora'),
    timestamp: options.timestamp || null,
  });
  const checkpointStep = Number.isInteger(Number(loraResult.lastCheckpoint?.step))
    ? Number(loraResult.lastCheckpoint.step)
    : null;
  const checkpointArtifacts = Array.isArray(loraResult.checkpointArtifacts)
    ? loraResult.checkpointArtifacts
    : [];
  const evalReports = Array.isArray(loraResult.evalReports) ? loraResult.evalReports : [];
  const exports = Array.isArray(loraResult.exports) ? loraResult.exports : [];
  const stageManifestPayload = {
    ...buildDistillArtifactBase(loadedWorkload, {
      prefix: 'dst_stage',
      artifactType: 'distill_stage_manifest',
      datasetPath: options.datasetPath || loadedWorkload.workload.datasetPath,
      datasetHash: resolveLoraDatasetHash(loraResult),
      stage: stageId,
      checkpointStep,
      configHash: loadedWorkload.workload.configHash,
      parentArtifacts: options.parentArtifacts || [],
    }),
    stageId,
    trainingStage: 'sft',
    objective: 'sft',
    stagePlanEntry: stageEntry,
    stepCount: checkpointStep,
    checkpointCount: checkpointArtifacts.length,
    bestCheckpointId: loraResult.lastCheckpoint?.id || checkpointArtifacts[checkpointArtifacts.length - 1]?.checkpointId || null,
    selectionMetric: stageEntry.selectionMetric,
    selectionGoal: stageEntry.selectionGoal,
    checkpointArtifacts,
    evalReports: evalReports.map((report) => ({
      checkpointId: report.checkpointId || null,
      evalDatasetId: report.evalDatasetId || null,
      reportPath: report.reportPath || null,
      primaryMetric: report.primaryMetric || null,
      primaryScore: report.primaryScore ?? null,
      loss: report.loss ?? null,
      baseline: report.baseline || null,
      qualityClaim: report.qualityClaim || null,
    })),
    exports: exports.map((entry) => ({
      checkpointId: entry.checkpointId || null,
      manifestPath: entry.manifestPath || null,
      runtimeManifestPath: entry.runtimeManifestPath || null,
      weightsPath: entry.weightsPath || null,
      weightsSha256: entry.weightsSha256 || null,
      exportPath: entry.exportPath || null,
    })),
    sftLoraRun: {
      runRoot: loraResult.runRoot,
      workloadId: loraResult.workloadId,
      runnerKind: loraResult.runnerKind,
      preflight: loraResult.preflight || null,
      baseModel: loraResult.baseModel || null,
      dataset: loraResult.dataset || null,
    },
    legacyArtifact: null,
    lastCheckpoint: loraResult.lastCheckpoint || null,
  };
  const stageManifest = await writeDistillStageManifest(options.layout, stageManifestPayload);
  return {
    stageId,
    trainingStage: 'sft',
    objective: 'sft',
    runnerKind: loraResult.runnerKind,
    metrics: loraResult.metrics,
    checkpointArtifacts,
    evalReports,
    exports,
    stageManifestPath: stageManifest.path,
    loraRunRoot: loraResult.runRoot,
    lastCheckpoint: loraResult.lastCheckpoint || null,
  };
}

async function runResolvedDistillStage(options) {
  const stageEntry = options.stageEntry;
  if (isSftDistillStage(stageEntry)) {
    return runDistillSftLoraStage(options);
  }
  if (String(stageEntry.trainingStage || stageEntry.id || '').includes('stage_b')) {
    return runDistillationStageB(options);
  }
  return runDistillationStageA(options);
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
  const workload = await loadWorkloadFromRunRoot(runRoot).then((loaded) => loaded.workload, () => null);
  const sorted = sortTrainingReports(await loadTrainingEvalReports(runRoot), workload);
  const payload = {
    artifactType: 'training_compare_report',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    runRoot,
    selectionMetric: workload?.selectionMetric || sorted[0]?.primaryMetric || null,
    selectionGoal: workload?.selectionGoal || null,
    count: sorted.length,
    best: sorted[0] || null,
    reports: sorted.map((report) => ({
      stage: report.stage || null,
      checkpointId: report.checkpointId || null,
      evalDatasetId: report.evalDatasetId || null,
      primaryMetric: report.primaryMetric || null,
      primaryScore: report.primaryScore ?? null,
      loss: report.loss ?? null,
      baseline: report.baseline || null,
      qualityClaim: report.qualityClaim || null,
      agentEval: report.agentEval || report.heldoutGate || null,
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
  const evalReports = await loadTrainingEvalReports(runRoot);
  const qualitySummary = summarizeQualityClaims(evalReports);
  const agentEvalSummary = summarizeAgentEvalReportRequirements(loadedWorkload.workload, evalReports);
  const requiredImprovementEvalIds = collectRequiredImprovementEvalIds(loadedWorkload.workload);
  if (evalReports.length > 0) {
    checks.push({
      name: 'eval_reports',
      path: runRoot,
      ok: true,
      count: evalReports.length,
    });
  }
  if (qualitySummary.count > 0) {
    checks.push({
      name: 'baseline_quality_claims',
      path: runRoot,
      ok: qualitySummary.failedRequiredCount === 0,
      ...qualitySummary,
    });
  }
  if (requiredImprovementEvalIds.length > 0 && qualitySummary.count === 0) {
    checks.push({
      name: 'required_improvement_claims',
      path: runRoot,
      ok: false,
      requiredEvalDatasetIds: requiredImprovementEvalIds,
      error: 'No baseline quality claims were written for eval datasets that require improvement.',
    });
  }
  if (agentEvalSummary.requiredCount > 0) {
    checks.push({
      name: 'agent_heldout_eval',
      path: runRoot,
      ok: agentEvalSummary.failedCount === 0,
      ...agentEvalSummary,
      error: agentEvalSummary.failedCount === 0
        ? null
        : 'One or more required agent held-out eval gates are missing or failing.',
    });
  }
  const passed = checks.every((entry) => entry.ok === true);
  const payload = {
    artifactType: 'training_quality_gate',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    runRoot,
    passed,
    qualitySummary,
    agentEvalSummary,
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
        const stageResult = await runResolvedDistillStage({
          loadedWorkload,
          stageEntry,
          layout: runArtifacts.layout,
          datasetPath,
          priorStageResult: stageResults[stageResults.length - 1] || null,
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
      const stageResult = await runResolvedDistillStage({
        loadedWorkload,
        stageEntry,
        layout: runArtifacts.layout,
        datasetPath,
        stageAArtifact: action === 'stage-b' ? request.stageArtifact || null : null,
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
        signal: request.signal ?? null,
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
    const result = await runLoraPipeline({
      loadedWorkload,
      runRoot: request.runRoot || null,
      timestamp: request.timestamp || null,
    });
    const compare = await compareLoraRun({
      runRoot: result.runRoot,
    });
    const qualityGate = await qualityGateLoraRun({
      runRoot: result.runRoot,
    });
    return {
      ...result,
      compare,
      qualityGate,
    };
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
        signal: request.signal ?? null,
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
