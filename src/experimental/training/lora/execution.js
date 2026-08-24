import { access, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { loadBackwardRegistry } from '../../../config/backward-registry-loader.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../memory/buffer-pool.js';
import { runMatmul } from '../../../gpu/kernels/index.js';
import { runResidualAdd } from '../../../gpu/kernels/residual.js';
import { parseJsonl } from '../datasets/jsonl.js';
import { loadTextPairsDataset, tokenizeTextPairs } from '../datasets/text-pairs.js';
import { LoraAdapter } from '../lora.js';
import { TrainingRunner, restoreTrainingCheckpointState } from '../runner.js';
import { AdamOptimizer } from '../optimizer.js';
import { crossEntropyLoss } from '../loss.js';
import { clipGradients } from '../clip.js';
import { OpType, AutogradTape } from '../autograd.js';
import { loadCheckpoint } from '../checkpoint.js';
import { exportLoRAAdapter } from '../export.js';
import { computeEvalMetrics } from '../operator-eval.js';
import { summarizeAgentEvalReportRequirements } from '../operator-agent-eval.js';
import { appendScoreboardRow } from '../operator-scoreboard.js';
import {
  buildArtifactBase,
  createTrainingRunLayout,
  hashArtifactPayload,
  writeJsonArtifact,
  writeRunContract,
  writeWorkloadLock,
} from '../operator-artifacts.js';
import { watchFinalizedCheckpoints } from '../checkpoint-watch.js';
import { loadLoRAFromManifest } from '../../adapters/lora-loader.js';
import { createUploadedTensor } from '../tensor-factory.js';
import { stableSortObject } from '../../../utils/stable-sort-object.js';
import { LORA_MODULE_ALIASES } from '../../../inference/pipelines/text/lora.js';
import { loadDistillModelHandle } from '../suite.js';
import { createDistillStudentRuntimeModelFixture } from '../distillation/student-fixture.js';
import { f16ToF32Array } from '../../../inference/kv-cache/types.js';
import { sha256BytesHex, sha256Hex } from '../../../utils/sha256.js';
import { LORA_RUNNER_BASE_MODEL_REGISTRY, finiteMetric, getLoraRunnerCompatibility, getPipelineConfig, isCausalLmLoraWorkload, isObjectRecord, normalizeLoraTargetModules, normalizeProviderEvalReport, preflightCausalLmLoraWorkload } from './plan.js';
import { buildArtifact, createCausalLmDatasetBatches, evaluateCausalLmLoraModel, evaluateToyLoraModel, exportCausalLmLoraModel, exportProviderCausalLmLoraModel, exportToyLoraModel, loadCausalLmTextPairSamples, loadToyLoraDataset, makeTensorFromFloat32, makeTensorFromUint32, releaseTensor, resolveDatasetPathForLoadedWorkload } from './checkpoint.js';

export function getCausalLmBaseModelRef(workload) {
  const pipeline = getPipelineConfig(workload);
  const baseModel = LORA_RUNNER_BASE_MODEL_REGISTRY[String(workload?.baseModelId || '')] || null;
  return String(
    pipeline.baseModelRef
    || workload.baseModelRef
    || workload.studentModelId
    || baseModel?.modelRef
    || workload.baseModelId
    || ''
  );
}

export function createLossGradient(loss, lossScale) {
  const lossElements = loss.shape.reduce((acc, value) => acc * value, 1);
  const gradData = new Float32Array(lossElements);
  gradData.fill(lossScale);
  return createUploadedTensor(gradData, 'f32', loss.shape, 'lora_causal_lm_loss_grad');
}

export function createCausalLmTrainingObjective() {
  return {
    name: 'causal_lm_cross_entropy',
    async forward({ model, batch, tape }) {
      if (typeof model.forwardCausalLm === 'function') {
        return model.forwardCausalLm(batch, tape);
      }
      const logits = await model.forward(batch.input, tape);
      return { logits };
    },
    async computeLoss({ batch, config, tape, forwardState }) {
      const loss = await crossEntropyLoss(forwardState.logits, batch.targets, config, tape);
      return {
        loss,
        components: {
          supervised_token_count: batch.supervisedTokenCount,
          ignored_target_count: batch.ignoredTargetCount,
        },
      };
    },
    backwardTargets({ batch, loss, lossScale }) {
      if (!Number.isInteger(batch.supervisedTokenCount) || batch.supervisedTokenCount < 1) {
        throw new Error('Causal-LM training batch requires supervisedTokenCount >= 1.');
      }
      return createLossGradient(loss, lossScale / batch.supervisedTokenCount);
    },
  };
}

export async function createCausalLmLoraFixture(workload) {
  const pipeline = getPipelineConfig(workload);
  const adapter = pipeline.adapter || {};
  const modelRef = getCausalLmBaseModelRef(workload);
  const handle = await loadDistillModelHandle(modelRef, 'lora base', {
    runtime: {
      shared: {
        debug: {
          logLevel: {
            defaultLogLevel: 'debug',
          },
        },
      },
      inference: {
        compute: {
          activationDtype: 'f32',
          keepF32Weights: true,
        },
      },
    },
  });
  let fixture = null;
  try {
    fixture = await createDistillStudentRuntimeModelFixture({
      training: {
        precision: workload.training?.precision || {},
      },
    }, {
      distillRuntime: {
        studentPipeline: handle.pipeline,
        studentGraphMode: 'transformer_full',
      },
      studentGraphMode: 'transformer_full',
      loraAdapter: {
        rank: adapter.rank,
        alpha: adapter.alpha,
        targetModules: normalizeLoraTargetModules(adapter),
      },
    });
  } catch (error) {
    if (handle.pipeline && typeof handle.pipeline.unload === 'function') {
      await handle.pipeline.unload();
    }
    throw error;
  }
  return {
    model: fixture.model,
    baseModelRef: modelRef,
    baseModelUrl: handle.modelUrl || null,
    baseManifest: handle.manifest || null,
    tokenizer: handle.pipeline.tokenizer,
    cleanup() {
      fixture.cleanup();
      if (handle.pipeline && typeof handle.pipeline.unload === 'function') {
        return handle.pipeline.unload();
      }
      return undefined;
    },
  };
}

export async function captureLoraParameterState(model) {
  const entries = typeof model?.loraTensorEntries === 'function'
    ? model.loraTensorEntries()
    : [];
  if (entries.length === 0) {
    throw new Error('Causal-LM LoRA parameter receipt requires adapter tensors.');
  }
  const tensors = [];
  for (const entry of [...entries].sort((left, right) => left.name.localeCompare(right.name))) {
    const tensor = entry.tensor;
    const elementCount = tensor.shape.reduce((product, value) => product * value, 1);
    const bytesPerElement = tensor.dtype === 'f16' ? 2 : 4;
    const raw = await readBuffer(tensor.buffer, elementCount * bytesPerElement);
    const values = tensor.dtype === 'f16'
      ? f16ToF32Array(new Uint16Array(raw))
      : new Float32Array(raw);
    let nonzeroCount = 0;
    let sumSquares = 0;
    let maxAbs = 0;
    for (const value of values) {
      if (value !== 0) nonzeroCount += 1;
      sumSquares += value * value;
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
    tensors.push({
      name: entry.name,
      dtype: tensor.dtype,
      shape: [...tensor.shape],
      elementCount,
      hash: sha256BytesHex(new Uint8Array(raw)),
      nonzeroCount,
      maxAbs,
      l2Norm: Math.sqrt(sumSquares),
      values,
    });
  }
  return {
    aggregateHash: sha256Hex(tensors.map((tensor) => `${tensor.name}\0${tensor.hash}\n`).join('')),
    tensors,
  };
}

export function buildLoraParameterReceipt(initial, final) {
  const initialByName = new Map(initial.tensors.map((tensor) => [tensor.name, tensor]));
  let totalSquaredDelta = 0;
  const tensors = final.tensors.map((finalTensor) => {
    const initialTensor = initialByName.get(finalTensor.name);
    if (!initialTensor || initialTensor.elementCount !== finalTensor.elementCount) {
      throw new Error(`LoRA parameter receipt shape mismatch for ${finalTensor.name}.`);
    }
    let squaredDelta = 0;
    let maxAbsDelta = 0;
    for (let index = 0; index < finalTensor.values.length; index += 1) {
      const delta = finalTensor.values[index] - initialTensor.values[index];
      squaredDelta += delta * delta;
      maxAbsDelta = Math.max(maxAbsDelta, Math.abs(delta));
    }
    totalSquaredDelta += squaredDelta;
    return {
      name: finalTensor.name,
      dtype: finalTensor.dtype,
      shape: finalTensor.shape,
      elementCount: finalTensor.elementCount,
      initialHash: initialTensor.hash,
      finalHash: finalTensor.hash,
      changed: initialTensor.hash !== finalTensor.hash,
      initialNonzeroCount: initialTensor.nonzeroCount,
      finalNonzeroCount: finalTensor.nonzeroCount,
      l2Delta: Math.sqrt(squaredDelta),
      maxAbsDelta,
    };
  });
  return {
    tensorCount: tensors.length,
    changedTensorCount: tensors.filter((tensor) => tensor.changed).length,
    nonzeroFinalTensorCount: tensors.filter((tensor) => tensor.finalNonzeroCount > 0).length,
    initialAggregateHash: initial.aggregateHash,
    finalAggregateHash: final.aggregateHash,
    aggregateChanged: initial.aggregateHash !== final.aggregateHash,
    l2Delta: Math.sqrt(totalSquaredDelta),
    tensors,
  };
}

export function getCausalLmFreezeConfig(workload) {
  const pipeline = getPipelineConfig(workload);
  const freeze = pipeline.freeze;
  if (!freeze || typeof freeze !== 'object' || Array.isArray(freeze)) {
    throw new Error('Causal-LM LoRA workload requires lora.freeze.');
  }
  return {
    encoder: freeze.encoder === true,
    prior: freeze.prior === true,
    decoder: freeze.decoder === true,
    base: freeze.base === true,
    lora: freeze.lora === true,
  };
}

export function createLoraRunnerTrainingConfig(workload, freeze) {
  return {
    training: {
      enabled: true,
      optimizer: {
        type: workload.training.optimizer.type,
        lr: workload.training.optimizer.lr,
        beta1: workload.training.optimizer.beta1,
        beta2: workload.training.optimizer.beta2,
        eps: workload.training.optimizer.eps,
        weightDecay: workload.training.optimizer.weightDecay,
        scheduler: workload.training.optimizer.scheduler,
      },
      gradient: {
        maxNorm: workload.training.gradientClipping.maxNorm,
      },
      precision: workload.training.precision,
      lossScaling: { enabled: false },
      distill: {
        enabled: false,
        stage: 'stage_a',
        teacherModelId: null,
        studentModelId: null,
        datasetId: null,
        datasetPath: null,
        languagePair: null,
        sourceLangs: null,
        targetLangs: null,
        pairAllowlist: null,
        strictPairContract: false,
        shardIndex: null,
        shardCount: null,
        resumeFrom: null,
        artifactDir: null,
        stageAArtifact: null,
        stageAArtifactHash: null,
        temperature: 1,
        alphaKd: 1,
        alphaCe: 0,
        allowHintFallback: false,
        tripletMargin: 0.2,
        studentGraphMode: 'transformer_full',
        freeze,
      },
      ul: {
        enabled: false,
        stage: 'stage1_joint',
        stage1Artifact: null,
        stage1ArtifactHash: null,
        artifactDir: null,
        lambda0: 5,
        seed: workload.seed,
        noiseSchedule: { name: 'linear', minSigma: 0.1, maxSigma: 1, steps: 1 },
        priorAlignment: { enabled: false, weight: 1 },
        decoderSigmoidWeight: { enabled: false, maxWeight: 1 },
        lossWeights: { prior: 1, decoder: 1, recon: 1 },
        freeze: null,
      },
    },
  };
}

export function createLoraOptimizer(workload) {
  return new AdamOptimizer({
    training: {
      optimizer: {
        type: workload.training.optimizer.type,
        lr: workload.training.optimizer.lr,
        beta1: workload.training.optimizer.beta1,
        beta2: workload.training.optimizer.beta2,
        eps: workload.training.optimizer.eps,
        weightDecay: workload.training.optimizer.weightDecay,
        scheduler: workload.training.optimizer.scheduler,
      },
      gradient: {
        maxNorm: workload.training.gradientClipping.maxNorm,
      },
      precision: workload.training.precision,
    },
  });
}

export function assertInternalCausalLmTrainerAllowed(workload, compatibility) {
  if (compatibility.observed.requiresExternalTrainer !== true) return;
  throw new Error(
    `Causal-LM LoRA base "${workload.baseModelId}" requires lora.trainer.modulePath or runLoraPipeline({ causalLmTrainer }) ` +
    'because the internal full-graph runner does not train packed q4k base weights.'
  );
}

export async function runInternalCausalLmLoraPipeline(options, layout, compatibility) {
  const loadedWorkload = options.loadedWorkload;
  const workload = loadedWorkload.workload;
  const pipeline = getPipelineConfig(workload);
  const datasetPath = await resolveDatasetPathForLoadedWorkload(workload.datasetPath, loadedWorkload);
  const preflight = await preflightCausalLmLoraWorkload(workload, {
    datasetPath,
    fetch: options.fetch,
    readFile: options.readFile,
  });
  if (!preflight.supported || preflight.blockedReasons.length > 0) {
    throw new Error(`Causal-LM LoRA workload is blocked: ${preflight.blockedReasons.join(',')}`);
  }
  assertInternalCausalLmTrainerAllowed(workload, compatibility);
  if (Math.floor(Number(workload.training.batchSize)) !== 1) {
    throw new Error('Causal-LM LoRA workload requires training.batchSize=1.');
  }
  const freeze = getCausalLmFreezeConfig(workload);
  const fixture = await createCausalLmLoraFixture(workload);
  try {
    const dataset = await loadCausalLmTextPairSamples(workload, datasetPath, fixture.tokenizer);
    const initialParameterState = await captureLoraParameterState(fixture.model);
    const evalReports = [];
    const checkpointArtifacts = [];
    const exports = [];
    const baselineEvalReports = await evaluateCausalLmLoraModel(workload, fixture, dataset, layout, {
      checkpointId: 'base-model',
      checkpointStep: 0,
      stage: 'base_model',
      configHash: workload.configHash,
      workloadPath: loadedWorkload.absolutePath,
      workloadSha256: loadedWorkload.workloadSha256,
    });
    for (const report of baselineEvalReports) {
      evalReports.push(report);
      await appendScoreboardRow(layout.scoreboard, {
        artifactType: 'training_scoreboard',
        schemaVersion: 1,
        generatedAt: new Date().toISOString(),
        checkpointId: report.checkpointId,
        checkpointStep: report.checkpointStep,
        stage: report.stage,
        evalDatasetId: report.evalDatasetId,
        primaryMetric: report.primaryMetric,
        primaryScore: report.primaryScore,
        loss: report.loss,
        metrics: {
          loss: report.loss,
          primaryScore: report.primaryScore,
        },
      }, {
        selectionMetric: workload.selectionMetric,
        selectionGoal: workload.selectionGoal,
      });
    }
    const baselineReportsByEvalDatasetId = Object.fromEntries(
      baselineEvalReports.map((report) => [report.evalDatasetId, report])
    );
    const finalizedCheckpointIds = new Set();
    const runner = new TrainingRunner(createLoraRunnerTrainingConfig(workload, freeze), {
      optimizer: createLoraOptimizer(workload),
      crossEntropyLoss,
      clipGradients,
      trainingObjective: createCausalLmTrainingObjective(),
      onCheckpoint: async (checkpoint) => {
        const checkpointId = `checkpoint-${String(checkpoint.step).padStart(6, '0')}`;
        if (finalizedCheckpointIds.has(checkpointId)) return;
        finalizedCheckpointIds.add(checkpointId);
        const checkpointPayload = {
          ...buildArtifact(loadedWorkload, {
            prefix: 'lora_ckpt',
            id: checkpointId,
            artifactType: 'training_checkpoint',
            datasetPath: dataset.absolutePath,
            datasetHash: dataset.datasetHash,
            checkpointStep: checkpoint.step,
          }),
          checkpointId,
          checkpointPath: checkpoint.path,
          optimizerStatePresent: true,
          schedulerStatePresent: workload.training.optimizer.scheduler.enabled === true,
          runnerKey: compatibility.observed.runnerKey,
          baseModelRef: fixture.baseModelRef,
          tokenization: dataset.tokenization,
          resumeLineage: checkpoint.metadata?.lineage || null,
        };
        await writeJsonArtifact(
          join(layout.checkpoints, checkpointId, 'checkpoint.json'),
          checkpointPayload
        );
        const checkpointArtifact = await writeJsonArtifact(
          join(layout.checkpoints, checkpointId, 'checkpoint.complete.json'),
          checkpointPayload
        );
        checkpointArtifacts.push({
          checkpointId,
          checkpointPath: checkpoint.path,
          markerPath: checkpointArtifact.path,
          checkpointStep: checkpoint.step,
        });
        if (pipeline.export?.enabled === true && pipeline.export.atCheckpoints === true) {
          exports.push(await exportCausalLmLoraModel(
            loadedWorkload,
            layout,
            fixture,
            checkpointId,
            checkpoint.step,
            dataset.datasetHash
          ));
        }
        const reports = await evaluateCausalLmLoraModel(workload, fixture, dataset, layout, {
          checkpointId,
          checkpointStep: checkpoint.step,
          configHash: workload.configHash,
          workloadPath: loadedWorkload.absolutePath,
          workloadSha256: loadedWorkload.workloadSha256,
          baselineReportsByEvalDatasetId,
        });
        for (const report of reports) {
          evalReports.push(report);
          await appendScoreboardRow(layout.scoreboard, {
            artifactType: 'training_scoreboard',
            schemaVersion: 1,
            generatedAt: new Date().toISOString(),
            checkpointId,
            checkpointStep: checkpoint.step,
            stage: report.stage,
            evalDatasetId: report.evalDatasetId,
            primaryMetric: report.primaryMetric,
            primaryScore: report.primaryScore,
            loss: report.loss,
            qualityClaim: report.qualityClaim || null,
            metrics: {
              loss: report.loss,
              primaryScore: report.primaryScore,
            },
          }, {
            selectionMetric: workload.selectionMetric,
            selectionGoal: workload.selectionGoal,
          });
        }
      },
    });
    const metrics = await runner.run(
      fixture.model,
      createCausalLmDatasetBatches(dataset.samples),
      {
        epochs: 1,
        batchSize: 1,
        shuffle: false,
        maxSteps: workload.training.steps,
        checkpointEvery: workload.checkpointEvery,
        checkpointKey: join(layout.checkpoints, 'latest.state.json'),
        modelId: workload.baseModelId,
        modelUrl: fixture.baseModelUrl,
        tokenizerHash: fixture.baseManifest?.tokenizerHash || null,
      }
    );
    const finalParameterState = await captureLoraParameterState(fixture.model);
    const parameterReceipt = buildLoraParameterReceipt(
      initialParameterState,
      finalParameterState
    );
    const finalCheckpointId = runner.lastCheckpoint
      ? `checkpoint-${String(runner.lastCheckpoint.step).padStart(6, '0')}`
      : null;
    if (
      pipeline.export?.enabled !== false
      && finalCheckpointId
      && exports.every((entry) => entry.checkpointId !== finalCheckpointId)
    ) {
      exports.push(await exportCausalLmLoraModel(
        loadedWorkload,
        layout,
        fixture,
        finalCheckpointId,
        runner.lastCheckpoint.step,
        dataset.datasetHash
      ));
    }
    return {
      ok: true,
      kind: 'lora',
      action: 'run',
      runnerKind: 'causal_lm_lora',
      workloadId: workload.id,
      runRoot: layout.runRoot,
      preflight,
      checkpointArtifacts,
      evalReports,
      exports,
      metrics,
      parameterReceipt,
      lastCheckpoint: runner.lastCheckpoint,
      dataset: {
        path: dataset.absolutePath,
        rowCount: dataset.rowCount,
        sampleCount: dataset.samples.length,
        datasetHash: dataset.datasetHash,
        tokenization: dataset.tokenization,
      },
      baseModel: {
        id: workload.baseModelId,
        ref: fixture.baseModelRef,
        url: fixture.baseModelUrl,
      },
    };
  } finally {
    await fixture.cleanup();
  }
}
