import { crossEntropyLoss as defaultCrossEntropyLoss } from '../loss.js';
import { createTrainingObjective } from './base.js';
import { readBuffer } from '../../../memory/buffer-pool.js';
import { f16ToF32Array, f32ToF16Array } from '../../../inference/kv-cache/types.js';
import { createUploadedTensor } from '../tensor-factory.js';

function toFinite(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function resolveDistillConfig(config) {
  return config?.training?.distill || {};
}

function allowHintFallback(config, batch) {
  const distill = resolveDistillConfig(config);
  return distill.allowHintFallback === true || batch?.distill?.allowHintFallback === true;
}

function isTensorLike(value) {
  return !!value
    && typeof value === 'object'
    && Array.isArray(value.shape)
    && value.buffer != null;
}

function createLossGradient(loss, lossScale) {
  const lossElements = loss.shape.reduce((acc, value) => acc * value, 1);
  const gradData = new Float32Array(lossElements);
  gradData.fill(lossScale);
  return createUploadedTensor(gradData, 'f32', loss.shape, 'distill_triplet_loss_grad_output');
}

function createGradientTensor(values, shape, dtype, label) {
  const floatValues = values instanceof Float32Array ? values : new Float32Array(values);
  const tensorDtype = dtype === 'f16' ? 'f16' : 'f32';
  const payload = tensorDtype === 'f16'
    ? f32ToF16Array(floatValues)
    : floatValues;
  return createUploadedTensor(payload, tensorDtype, shape, label);
}

async function readLogitsRows(logitsTensor) {
  if (!isTensorLike(logitsTensor) || logitsTensor.shape.length < 2) {
    throw new Error('Distill triplet objective requires logits tensor with shape [batch, dim].');
  }
  const rows = Math.max(1, Math.floor(Number(logitsTensor.shape[0]) || 0));
  const cols = Math.max(1, Math.floor(Number(logitsTensor.shape[1]) || 0));
  const raw = await readBuffer(logitsTensor.buffer);
  const flat = logitsTensor.dtype === 'f16'
    ? f16ToF32Array(new Uint16Array(raw))
    : new Float32Array(raw);
  const requiredSize = rows * cols;
  if (flat.length < requiredSize) {
    throw new Error(
      `Distill triplet objective logits readback underflow: expected ${requiredSize}, got ${flat.length}.`
    );
  }
  const slices = [];
  for (let row = 0; row < rows; row += 1) {
    const start = row * cols;
    const end = start + cols;
    slices.push(flat.subarray(start, end));
  }
  return { rows, cols, slices };
}

function resolveTripletMask(batch, rowCount) {
  const mask = Array.isArray(batch?.distill?.tripletMask) ? batch.distill.tripletMask : null;
  if (!mask) {
    return new Array(rowCount).fill(true);
  }
  const resolved = new Array(rowCount);
  for (let row = 0; row < rowCount; row += 1) {
    resolved[row] = mask[row] !== false;
  }
  return resolved;
}

function resolveRowTensorCount(...entries) {
  let rowCount = Number.POSITIVE_INFINITY;
  for (const entry of entries) {
    const rows = Math.max(0, Math.floor(Number(entry?.rows) || 0));
    rowCount = Math.min(rowCount, rows);
  }
  return Number.isFinite(rowCount) ? rowCount : 0;
}

function gatherTripletRow(values, tokenIndices, expectedSize) {
  const size = Math.max(1, Math.floor(expectedSize));
  const output = new Float32Array(size);
  for (let i = 0; i < size; i += 1) {
    const tokenIndex = Array.isArray(tokenIndices) ? Number(tokenIndices[i]) : NaN;
    if (Number.isInteger(tokenIndex) && tokenIndex >= 0 && tokenIndex < values.length) {
      output[i] = values[tokenIndex];
      continue;
    }
    if (i < values.length) {
      output[i] = values[i];
      continue;
    }
    output[i] = 0;
  }
  return output;
}

function mapTripletGradIndex(tokenIndices, logicalIndex, rowCols) {
  const tokenIndex = Array.isArray(tokenIndices) ? Number(tokenIndices[logicalIndex]) : NaN;
  if (Number.isInteger(tokenIndex) && tokenIndex >= 0 && tokenIndex < rowCols) {
    return tokenIndex;
  }
  return logicalIndex < rowCols ? logicalIndex : -1;
}

function extractLogitsFromForwardOutput(output) {
  if (isTensorLike(output)) return output;
  if (isTensorLike(output?.logits)) return output.logits;
  return null;
}

async function resolveForwardLogits(model, batch, tape, phase) {
  if (model && typeof model.forwardDistill === 'function') {
    const output = await model.forwardDistill(batch, tape, { phase, stage: 'stage_b' });
    const logits = extractLogitsFromForwardOutput(output);
    if (!logits) {
      throw new Error(`Distill triplet objective: model.forwardDistill() returned invalid ${phase} logits.`);
    }
    return logits;
  }
  if (phase === 'positive') {
    return model.forward(batch.tripletPositiveInput, tape);
  }
  if (phase === 'negative') {
    return model.forward(batch.tripletNegativeInput, tape);
  }
  return model.forward(batch.input, tape);
}

export function createDistillTripletObjective(options = {}) {
  const lossFn = options.crossEntropyLoss || defaultCrossEntropyLoss;
  if (typeof lossFn !== 'function') {
    throw new Error('Distill triplet objective requires crossEntropyLoss(logits, targets, config, tape).');
  }

  return createTrainingObjective({
    name: 'triplet',
    async prepareBatch({ batch, config, options: runOptions }) {
      const distill = resolveDistillConfig(config);
      if (distill.stage !== 'stage_b') {
        throw new Error('Distill triplet objective requires training.distill.stage="stage_b".');
      }
      if (!distill.stageAArtifact) {
        throw new Error('Distill triplet objective requires training.distill.stageAArtifact.');
      }
      const stageAContext = runOptions?.stageAArtifactContext;
      if (!stageAContext || typeof stageAContext !== 'object') {
        throw new Error('Distill triplet objective requires stageAArtifactContext.');
      }
      const hasPromptTriplets = Array.isArray(batch?.distill?.tripletPositivePrompts)
        && Array.isArray(batch?.distill?.tripletNegativePrompts);
      const hasTensorTriplets = isTensorLike(batch?.tripletPositiveInput) && isTensorLike(batch?.tripletNegativeInput);
      if (!hasPromptTriplets && !hasTensorTriplets) {
        throw new Error(
          'Distill triplet objective requires per-step positive/negative triplet inputs (no precomputed hint fallback).'
        );
      }
      return batch;
    },
    async forward({ model, batch, tape }) {
      const logits = await resolveForwardLogits(model, batch, tape, 'anchor');
      const positiveLogits = await resolveForwardLogits(model, batch, tape, 'positive');
      const negativeLogits = await resolveForwardLogits(model, batch, tape, 'negative');
      return { logits, positiveLogits, negativeLogits };
    },
    async computeLoss({ batch, config, tape, forwardState, options: runOptions }) {
      const loss = await lossFn(forwardState.logits, batch.targets, config, tape);
      const distill = resolveDistillConfig(config);
      const margin = Math.max(0, toFinite(batch?.distill?.tripletMargin, toFinite(distill.tripletMargin, 0.2)));
      const stageAContext = runOptions?.stageAArtifactContext;
      const referenceKdMean = toFinite(stageAContext?.metricsSummary?.kdMean, 0.08);
      const hintFallbackAllowed = allowHintFallback(config, batch);
      let anchorRows = null;
      let positiveRows = null;
      let negativeRows = null;
      try {
        anchorRows = await readLogitsRows(forwardState.logits);
        positiveRows = await readLogitsRows(forwardState.positiveLogits);
        negativeRows = await readLogitsRows(forwardState.negativeLogits);
      } catch (error) {
        if (!hintFallbackAllowed) {
          throw error;
        }
        const tripletValues = Array.isArray(batch?.distill?.tripletLossValues)
          ? batch.distill.tripletLossValues
          : [];
        if (tripletValues.length <= 0) {
          throw error;
        }
        let tripletSumFallback = 0;
        let tripletCountFallback = 0;
        for (const value of tripletValues) {
          const normalized = toFinite(value, NaN);
          if (!Number.isFinite(normalized)) continue;
          tripletSumFallback += normalized;
          tripletCountFallback += 1;
        }
        const fallbackMean = tripletCountFallback > 0
          ? (tripletSumFallback / tripletCountFallback)
          : toFinite(batch?.distill?.tripletLossMean, 0);
        const lossTripletFallback = Math.max(0, fallbackMean + margin);
        const teacherModelIdFallback = batch?.distill?.teacherModelId || distill.teacherModelId || null;
        const studentModelIdFallback = batch?.distill?.studentModelId || distill.studentModelId || null;
        return {
          loss,
          components: {
            loss_triplet: lossTripletFallback,
            distill_stage: 'stage_b',
            distill_triplet_margin: margin,
            distill_triplet_active_count: tripletCountFallback,
            distill_stage_a_step_count: toFinite(stageAContext?.metricsSummary?.stepCount, 0),
            distill_stage_a_kd_mean: referenceKdMean,
            distill_teacher_model_id: teacherModelIdFallback,
            distill_student_model_id: studentModelIdFallback,
          },
          _distillBackward: null,
        };
      }
      const rowCount = resolveRowTensorCount(anchorRows, positiveRows, negativeRows);
      if (rowCount <= 0) {
        throw new Error('Distill triplet objective requires non-empty anchor/positive/negative logits.');
      }
      const teacherTokenRows = Array.isArray(batch?.distill?.teacherTopTokenIndices)
        ? batch.distill.teacherTopTokenIndices
        : [];
      const fallbackDim = Math.min(anchorRows.cols, positiveRows.cols, negativeRows.cols);
      const hintedDim = Math.floor(Number(batch?.distill?.teacherTopProbs?.[0]?.length) || 0);
      const dim = hintedDim > 0 ? hintedDim : fallbackDim;
      if (dim <= 0) {
        throw new Error('Distill triplet objective requires non-empty logits dimension.');
      }

      const mask = resolveTripletMask(batch, rowCount);
      const anchorGrad = new Float32Array(anchorRows.rows * anchorRows.cols);
      const positiveGrad = new Float32Array(positiveRows.rows * positiveRows.cols);
      const negativeGrad = new Float32Array(negativeRows.rows * negativeRows.cols);
      let consideredRows = 0;
      let tripletSum = 0;
      let activeTriplets = 0;

      for (let row = 0; row < rowCount; row += 1) {
        if (!mask[row]) continue;
        consideredRows += 1;
        const teacherTokens = Array.isArray(teacherTokenRows[row]) ? teacherTokenRows[row] : [];
        const anchor = gatherTripletRow(anchorRows.slices[row], teacherTokens, dim);
        const positive = gatherTripletRow(positiveRows.slices[row], teacherTokens, dim);
        const negative = gatherTripletRow(negativeRows.slices[row], teacherTokens, dim);

        let dPos = 0;
        let dNeg = 0;
        for (let col = 0; col < dim; col += 1) {
          const ap = anchor[col] - positive[col];
          const an = anchor[col] - negative[col];
          dPos += ap * ap;
          dNeg += an * an;
        }
        dPos /= dim;
        dNeg /= dim;
        const hinge = margin + dPos - dNeg;
        const rowLoss = Math.max(0, hinge);
        tripletSum += rowLoss;
        if (rowLoss <= 0) continue;
        activeTriplets += 1;

        const anchorOffset = row * anchorRows.cols;
        const positiveOffset = row * positiveRows.cols;
        const negativeOffset = row * negativeRows.cols;
        for (let col = 0; col < dim; col += 1) {
          const a = anchor[col];
          const p = positive[col];
          const n = negative[col];
          const mappedAnchorCol = mapTripletGradIndex(teacherTokens, col, anchorRows.cols);
          const mappedPositiveCol = mapTripletGradIndex(teacherTokens, col, positiveRows.cols);
          const mappedNegativeCol = mapTripletGradIndex(teacherTokens, col, negativeRows.cols);
          if (mappedAnchorCol >= 0) {
            anchorGrad[anchorOffset + mappedAnchorCol] += (2 * (n - p)) / dim;
          }
          if (mappedPositiveCol >= 0) {
            positiveGrad[positiveOffset + mappedPositiveCol] += (2 * (p - a)) / dim;
          }
          if (mappedNegativeCol >= 0) {
            negativeGrad[negativeOffset + mappedNegativeCol] += (2 * (a - n)) / dim;
          }
        }
      }

      const normalizer = consideredRows > 0 ? consideredRows : 1;
      for (let i = 0; i < anchorGrad.length; i += 1) {
        anchorGrad[i] /= normalizer;
      }
      for (let i = 0; i < positiveGrad.length; i += 1) {
        positiveGrad[i] /= normalizer;
      }
      for (let i = 0; i < negativeGrad.length; i += 1) {
        negativeGrad[i] /= normalizer;
      }
      const lossTriplet = tripletSum / normalizer;

      const teacherModelId = batch?.distill?.teacherModelId || distill.teacherModelId || null;
      const studentModelId = batch?.distill?.studentModelId || distill.studentModelId || null;
      const anchorDtype = forwardState.logits?.dtype;
      const positiveDtype = forwardState.positiveLogits?.dtype;
      const negativeDtype = forwardState.negativeLogits?.dtype;
      if (anchorDtype === undefined || positiveDtype === undefined || negativeDtype === undefined) {
        throw new Error('distill_triplet: forwardState logits dtype is required for all three branches (anchor/positive/negative)');
      }
      return {
        loss,
        components: {
          loss_triplet: lossTriplet,
          distill_stage: 'stage_b',
          distill_triplet_margin: margin,
          distill_triplet_active_count: activeTriplets,
          distill_stage_a_step_count: toFinite(stageAContext?.metricsSummary?.stepCount, 0),
          distill_stage_a_kd_mean: referenceKdMean,
          distill_teacher_model_id: teacherModelId,
          distill_student_model_id: studentModelId,
        },
        _distillBackward: {
          anchor: {
            shape: [anchorRows.rows, anchorRows.cols],
            dtype: anchorDtype,
            gradValues: anchorGrad,
          },
          positive: {
            shape: [positiveRows.rows, positiveRows.cols],
            dtype: positiveDtype,
            gradValues: positiveGrad,
          },
          negative: {
            shape: [negativeRows.rows, negativeRows.cols],
            dtype: negativeDtype,
            gradValues: negativeGrad,
          },
        },
      };
    },
    backwardTargets({ loss, lossScale, lossResult, forwardState }) {
      const seeds = [];
      const ceSeed = createLossGradient(loss, lossScale);
      seeds.push({ tensor: loss, grad: ceSeed });
      const distillBackward = lossResult?._distillBackward;
      if (distillBackward && isTensorLike(forwardState?.logits)) {
        const anchorSeed = createGradientTensor(
          distillBackward.anchor.gradValues,
          distillBackward.anchor.shape,
          distillBackward.anchor.dtype,
          'distill_triplet_anchor_grad_output'
        );
        seeds.push({ tensor: forwardState.logits, grad: anchorSeed });
      }
      if (distillBackward && isTensorLike(forwardState?.positiveLogits)) {
        const positiveSeed = createGradientTensor(
          distillBackward.positive.gradValues,
          distillBackward.positive.shape,
          distillBackward.positive.dtype,
          'distill_triplet_positive_grad_output'
        );
        seeds.push({ tensor: forwardState.positiveLogits, grad: positiveSeed });
      }
      if (distillBackward && isTensorLike(forwardState?.negativeLogits)) {
        const negativeSeed = createGradientTensor(
          distillBackward.negative.gradValues,
          distillBackward.negative.shape,
          distillBackward.negative.dtype,
          'distill_triplet_negative_grad_output'
        );
        seeds.push({ tensor: forwardState.negativeLogits, grad: negativeSeed });
      }
      return { seeds };
    },
    metrics({ config, lossResult }) {
      const distill = resolveDistillConfig(config);
      const components = lossResult.components || {};
      return {
        loss_triplet: Number.isFinite(components.loss_triplet) ? components.loss_triplet : 0,
        distill_stage: 'stage_b',
        distill_triplet_margin: Number.isFinite(components.distill_triplet_margin)
          ? components.distill_triplet_margin
          : toFinite(distill.tripletMargin, 0.2),
        distill_triplet_active_count: Number.isFinite(components.distill_triplet_active_count)
          ? components.distill_triplet_active_count
          : 0,
        distill_stage_a_step_count: Number.isFinite(components.distill_stage_a_step_count)
          ? components.distill_stage_a_step_count
          : 0,
        distill_stage_a_kd_mean: Number.isFinite(components.distill_stage_a_kd_mean)
          ? components.distill_stage_a_kd_mean
          : null,
        distill_teacher_model_id: typeof components.distill_teacher_model_id === 'string'
          ? components.distill_teacher_model_id
          : (distill.teacherModelId || null),
        distill_student_model_id: typeof components.distill_student_model_id === 'string'
          ? components.distill_student_model_id
          : (distill.studentModelId || null),
      };
    },
    cleanup({ model }) {
      if (model && typeof model.cleanupDistillStep === 'function') {
        model.cleanupDistillStep();
      }
    },
  });
}
