import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

import { loadBackwardRegistry } from '../../../config/backward-registry-loader.js';
import { f16ToF32Array } from '../../../inference/kv-cache/types.js';
import { readBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { AutogradTape } from '../autograd.js';
import { loadCheckpoint } from '../checkpoint.js';
import { computeEvalMetrics } from '../operator-eval.js';
import {
  buildDistillPrompt,
  createDistillRuntimeContext,
  createDistillStudentRuntimeModelFixture,
  resolveDistillDataScope,
} from '../suite.js';
import { restoreTrainingCheckpointState } from '../runner.js';
import { loadCanonicalTranslationDataset } from './dataset.js';
import { buildDistillArtifactBase, writeDistillEvalReport } from './artifacts.js';
import { buildDistillationTrainingConfigFromWorkload, resolveInternalDistillStage } from './runtime.js';

function toFloat32Array(raw, dtype = 'f32') {
  if (dtype === 'f16') {
    return f16ToF32Array(new Uint16Array(raw));
  }
  return new Float32Array(raw);
}

function resolveEvalDatasets(workload, requestedEvalDatasetId = null) {
  const evalDatasets = Array.isArray(workload.evalDatasets) ? workload.evalDatasets : [];
  if (!requestedEvalDatasetId) {
    return evalDatasets;
  }
  return evalDatasets.filter((entry) => entry.id === requestedEvalDatasetId);
}

function argmax(values) {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    const value = Number.isFinite(values[index]) ? values[index] : Number.NEGATIVE_INFINITY;
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function releaseTensorLike(value, released) {
  if (!value || typeof value !== 'object') return;
  const buffer = value.buffer;
  if (!buffer || released.has(buffer)) return;
  released.add(buffer);
  releaseBuffer(buffer);
}

function disposeTapeOutputs(tape, protectedBuffers = new Set()) {
  if (!tape || !Array.isArray(tape.records)) return;
  const released = new Set();
  for (const record of tape.records) {
    const output = record?.output;
    if (!output || typeof output !== 'object') continue;
    if (output.buffer && !protectedBuffers.has(output.buffer)) {
      releaseTensorLike(output, released);
      continue;
    }
    if (Array.isArray(output)) {
      for (const entry of output) {
        if (entry?.buffer && !protectedBuffers.has(entry.buffer)) {
          releaseTensorLike(entry, released);
        }
      }
    }
  }
}

function collectProtectedBuffers(model) {
  const protectedBuffers = new Set();
  const groups = typeof model?.paramGroups === 'function'
    ? model.paramGroups()
    : {};
  for (const params of Object.values(groups || {})) {
    for (const tensor of Array.isArray(params) ? params : []) {
      if (tensor?.buffer) {
        protectedBuffers.add(tensor.buffer);
      }
    }
  }
  return protectedBuffers;
}

async function readLogitsTensor(tensor) {
  if (tensor.dtype === undefined) {
    throw new Error('readLogitsTensor: tensor.dtype is required');
  }
  const raw = await readBuffer(tensor.buffer);
  return toFloat32Array(raw, tensor.dtype);
}

async function greedyDecodeFixture(model, tokenizer, prompt, decodePolicy = {}) {
  const maxTokens = Number.isInteger(decodePolicy?.maxTokens) && decodePolicy.maxTokens > 0
    ? decodePolicy.maxTokens
    : null;
  if (!maxTokens) {
    throw new Error('Translation eval requires evalDatasets[].decodePolicy.maxTokens in the workload pack.');
  }
  const stopOnEos = decodePolicy?.stopOnEos !== false;
  const eosToken = tokenizer?.getSpecialTokens?.()?.eos ?? null;
  const protectedBuffers = collectProtectedBuffers(model);
  const generated = [];
  let currentPrompt = prompt;
  for (let step = 0; step < maxTokens; step += 1) {
    const tape = new AutogradTape(loadBackwardRegistry());
    let logits = null;
    try {
      const result = await model.forwardDistill(
        {
          distill: {
            prompts: [currentPrompt],
          },
        },
        tape,
        { phase: 'anchor' }
      );
      logits = result?.logits || result;
      const values = await readLogitsTensor(logits);
      const tokenId = argmax(values);
      if (stopOnEos && eosToken != null && tokenId === eosToken) {
        break;
      }
      generated.push(tokenId);
      currentPrompt = `${prompt}${tokenizer.decode(generated, false, false)}`;
    } finally {
      if (logits?.buffer && !protectedBuffers.has(logits.buffer)) {
        releaseBuffer(logits.buffer);
      }
      model.cleanupDistillStep?.();
      disposeTapeOutputs(tape, protectedBuffers);
    }
  }
  return tokenizer.decode(generated, true, true);
}

function flattenMetricSummary(metrics) {
  return {
    bleu: metrics?.bleu?.score ?? null,
    chrf: metrics?.chrf?.score ?? null,
    exact_match: metrics?.exactMatch?.score ?? null,
    accuracy: metrics?.accuracy?.score ?? null,
    primaryMetric: metrics?.primaryMetric || null,
    primaryScore: metrics?.primaryScore ?? null,
  };
}

export async function evaluateDistillationModel(options) {
  const loadedWorkload = options.loadedWorkload;
  const workload = loadedWorkload.workload;
  const stageId = options.stageId;
  const checkpointId = options.checkpointId;
  const checkpointStep = options.checkpointStep;
  const distillRuntime = options.distillRuntime;
  const model = options.model;
  if (distillRuntime.studentGraphMode !== 'transformer_full') {
    throw new Error(
      `Distillation eval requires studentGraphMode="transformer_full"; got "${distillRuntime.studentGraphMode}".`
    );
  }
  const evalDatasets = resolveEvalDatasets(workload, options.evalDatasetId || null);
  if (evalDatasets.length === 0) {
    throw new Error(`No eval datasets resolved for workload "${workload.id}".`);
  }
  const reports = [];
  for (const evalDataset of evalDatasets) {
    if (evalDataset.evalKind !== 'translation') {
      throw new Error(`Distillation eval currently supports translation eval only, got "${evalDataset.evalKind}".`);
    }
    const dataset = await loadCanonicalTranslationDataset(evalDataset.datasetPath, {
      strictPairContract: workload.pipeline.strictPairContract === true,
      sourceLangs: evalDataset.sourceLangs || workload.pipeline.sourceLangs,
      targetLangs: evalDataset.targetLangs || workload.pipeline.targetLangs,
      pairAllowlist: evalDataset.pairAllowlist || workload.pipeline.pairAllowlist,
    });
    const hypotheses = [];
    const references = [];
    const samples = [];
    for (const row of dataset.rows) {
      const prompt = buildDistillPrompt({
        direction: row.pair || (
          row.src_lang && row.tgt_lang
            ? `${row.src_lang}->${row.tgt_lang}`
            : 'unknown'
        ),
        source: row.source,
      });
      const hypothesis = await greedyDecodeFixture(
        model,
        distillRuntime.studentPipeline.tokenizer,
        prompt,
        evalDataset.decodePolicy || {}
      );
      hypotheses.push(hypothesis);
      references.push(row.target_pos);
      if (samples.length < 5) {
        samples.push({
          row_id: row.row_id,
          source: row.source,
          reference: row.target_pos,
          hypothesis,
        });
      }
    }
    const computedMetrics = computeEvalMetrics('translation', hypotheses, references, {});
    const flattened = flattenMetricSummary(computedMetrics);
    const reportPayload = {
      ...buildDistillArtifactBase(loadedWorkload, {
        prefix: 'dst_eval',
        artifactType: 'training_eval_report',
        datasetPath: dataset.absolutePath,
        datasetHash: dataset.canonicalHash,
        stage: stageId,
        checkpointStep,
        parentArtifacts: options.parentArtifacts || [],
        configHash: options.configHash || workload.configHash,
      }),
      checkpointId,
      checkpointPath: options.checkpointPath || null,
      evalDatasetId: evalDataset.id,
      evalKind: evalDataset.evalKind,
      metrics: computedMetrics,
      ...flattened,
      rowCount: dataset.rowCount,
      sampleRows: samples,
    };
    const reportFile = options.layout
      ? await writeDistillEvalReport(options.layout, reportPayload)
      : null;
    reports.push({
      ...reportPayload,
      reportPath: reportFile?.path || null,
    });
  }
  return reports;
}

export async function evaluateDistillationCheckpoint(options) {
  const loadedWorkload = options.loadedWorkload;
  const workload = loadedWorkload.workload;
  const stagePlan = workload.pipeline.stagePlan;
  const requestedStageId = String(options.stageId || '').trim();
  const stageEntry = stagePlan.find((entry) => entry.id === requestedStageId)
    || stagePlan[0];
  if (!stageEntry) {
    throw new Error(`No stage entry resolved for workload "${workload.id}".`);
  }
  const internalStage = resolveInternalDistillStage(stageEntry);
  const checkpointPath = resolve(String(options.checkpointPath));
  const checkpointRecord = await loadCheckpoint(checkpointPath);
  if (!checkpointRecord) {
    throw new Error(`Checkpoint not found: ${checkpointPath}`);
  }
  const configBundle = buildDistillationTrainingConfigFromWorkload(loadedWorkload, stageEntry, {
    datasetPath: options.datasetPath || workload.datasetPath,
    stageAArtifact: options.stageAArtifact || null,
    stageAArtifactHash: options.stageAArtifactHash || null,
    artifactDir: dirname(dirname(checkpointPath)),
  });
  const distillRuntime = await createDistillRuntimeContext({
    teacherModelId: workload.teacherModelId,
    studentModelId: workload.studentModelId,
    trainingStage: internalStage,
    studentGraphMode: workload.pipeline.studentGraphMode,
  }, configBundle.trainingConfig.training);
  let fixture = null;
  try {
    fixture = await createDistillStudentRuntimeModelFixture({
      training: configBundle.trainingConfig.training,
    }, {
      distillRuntime,
      studentGraphMode: workload.pipeline.studentGraphMode,
    });
    await restoreTrainingCheckpointState(
      fixture.model,
      { getState: () => null, stepCount: 0 },
      checkpointRecord,
      fixture.config
    );
    return evaluateDistillationModel({
      loadedWorkload,
      layout: options.layout || null,
      stageId: stageEntry.id,
      checkpointId: options.checkpointId || 'checkpoint',
      checkpointStep: options.checkpointStep || null,
      checkpointPath,
      distillRuntime,
      model: fixture.model,
      evalDatasetId: options.evalDatasetId || null,
      configHash: configBundle.trainingConfigHash,
      parentArtifacts: options.parentArtifacts || [],
    });
  } finally {
    fixture?.cleanup?.();
    await distillRuntime.cleanup();
  }
}

export async function readDistillCheckpointMarker(markerPath) {
  const absolutePath = resolve(String(markerPath));
  const raw = await readFile(absolutePath, 'utf8');
  return {
    absolutePath,
    marker: JSON.parse(raw),
  };
}
