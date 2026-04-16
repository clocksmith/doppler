import { crossEntropyLoss as defaultCrossEntropyLoss } from '../loss.js';
import { createTrainingObjective } from './base.js';
import { readBuffer } from '../../memory/buffer-pool.js';
import { f16ToF32Array, f32ToF16Array } from '../../inference/kv-cache/types.js';
import { createUploadedTensor } from '../tensor-factory.js';

const EPS = 1e-8;

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
  return createUploadedTensor(gradData, 'f32', loss.shape, 'distill_kd_loss_grad_output');
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
    throw new Error('Distill KD objective requires logits tensor with shape [batch, dim].');
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
      `Distill KD objective logits readback underflow: expected ${requiredSize}, got ${flat.length}.`
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

function softmax(values, temperature = 1) {
  const t = Math.max(1e-4, toFinite(temperature, 1));
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const candidate = values[i] / t;
    if (candidate > max) max = candidate;
  }
  const exps = new Float32Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const value = Math.exp((values[i] / t) - max);
    exps[i] = value;
    sum += value;
  }
  if (!Number.isFinite(sum) || sum <= 0) {
    const uniform = 1 / Math.max(1, values.length);
    exps.fill(uniform);
    return exps;
  }
  for (let i = 0; i < exps.length; i += 1) {
    exps[i] /= sum;
  }
  return exps;
}

function normalizeProbRow(values, expectedSize) {
  const output = new Float32Array(expectedSize);
  if (Array.isArray(values) || ArrayBuffer.isView(values)) {
    const source = Array.isArray(values) ? values : Array.from(values);
    const count = Math.min(expectedSize, source.length);
    for (let i = 0; i < count; i += 1) {
      output[i] = Math.max(0, toFinite(source[i], 0));
    }
  }
  let sum = 0;
  for (let i = 0; i < output.length; i += 1) {
    sum += output[i];
  }
  if (!Number.isFinite(sum) || sum <= 0) {
    const uniform = 1 / Math.max(1, output.length);
    output.fill(uniform);
    return output;
  }
  for (let i = 0; i < output.length; i += 1) {
    output[i] /= sum;
  }
  return output;
}

function gatherStudentLogitsRow(logitsRow, tokenIndices, expectedSize) {
  const size = Math.max(1, Math.floor(expectedSize));
  const gathered = new Float32Array(size);
  for (let i = 0; i < size; i += 1) {
    const tokenIndex = Array.isArray(tokenIndices) ? Number(tokenIndices[i]) : NaN;
    if (Number.isInteger(tokenIndex) && tokenIndex >= 0 && tokenIndex < logitsRow.length) {
      gathered[i] = logitsRow[tokenIndex];
      continue;
    }
    if (i < logitsRow.length) {
      gathered[i] = logitsRow[i];
      continue;
    }
    gathered[i] = DISTILL_LOGIT_FALLBACK;
  }
  return gathered;
}

function klDivergence(teacherProbs, studentProbs) {
  const size = Math.min(teacherProbs.length, studentProbs.length);
  if (size <= 0) return 0;
  let total = 0;
  for (let i = 0; i < size; i += 1) {
    const p = Math.max(EPS, teacherProbs[i]);
    const q = Math.max(EPS, studentProbs[i]);
    total += p * (Math.log(p) - Math.log(q));
  }
  return total;
}

function argmax(values) {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const value = Number.isFinite(values[i]) ? values[i] : Number.NEGATIVE_INFINITY;
    if (value > bestValue) {
      bestValue = value;
      bestIndex = i;
    }
  }
  return bestIndex;
}

async function resolveForwardLogits(model, batch, tape) {
  if (model && typeof model.forwardDistill === 'function') {
    const output = await model.forwardDistill(batch, tape, { phase: 'anchor', stage: 'stage_a' });
    if (isTensorLike(output)) {
      return output;
    }
    if (isTensorLike(output?.logits)) {
      return output.logits;
    }
    throw new Error('Distill KD objective: model.forwardDistill() must return a logits tensor.');
  }
  return model.forward(batch.input, tape);
}

export function createDistillKdObjective(options = {}) {
  const lossFn = options.crossEntropyLoss || defaultCrossEntropyLoss;
  if (typeof lossFn !== 'function') {
    throw new Error('Distill KD objective requires crossEntropyLoss(logits, targets, config, tape).');
  }

  return createTrainingObjective({
    name: 'kd',
    async forward({ model, batch, tape }) {
      const logits = await resolveForwardLogits(model, batch, tape);
      return { logits };
    },
    async computeLoss({ batch, config, tape, forwardState }) {
      const loss = await lossFn(forwardState.logits, batch.targets, config, tape);
      const distill = resolveDistillConfig(config);
      const temperature = Math.max(1e-4, toFinite(batch?.distill?.temperature, toFinite(distill.temperature, 1)));
      const alphaKd = toFinite(batch?.distill?.alphaKd, toFinite(distill.alphaKd, 1));
      const alphaCe = toFinite(batch?.distill?.alphaCe, toFinite(distill.alphaCe, 0));
      const teacherRowsRaw = Array.isArray(batch?.distill?.teacherTopProbs)
        ? batch.distill.teacherTopProbs
        : null;
      if (!teacherRowsRaw || teacherRowsRaw.length === 0) {
        throw new Error('Distill KD objective requires batch.distill.teacherTopProbs from teacher logits.');
      }

      const hintFallbackAllowed = allowHintFallback(config, batch);
      let logitsRows = null;
      try {
        logitsRows = await readLogitsRows(forwardState.logits);
      } catch (error) {
        logitsRows = null;
        if (
          !hintFallbackAllowed
          || !Array.isArray(batch?.distill?.studentTopProbs)
          || batch.distill.studentTopProbs.length === 0
        ) {
          throw error;
        }
      }
      const teacherTargets = Array.isArray(batch?.distill?.teacherTargetIndices)
        ? batch.distill.teacherTargetIndices
        : [];
      const teacherTargetTokenIds = Array.isArray(batch?.distill?.teacherTargetTokenIds)
        ? batch.distill.teacherTargetTokenIds
        : [];
      const teacherTokenRows = Array.isArray(batch?.distill?.teacherTopTokenIndices)
        ? batch.distill.teacherTopTokenIndices
        : [];
      const studentProbRowsFallback = Array.isArray(batch?.distill?.studentTopProbs)
        ? batch.distill.studentTopProbs
        : [];
      const rowCount = logitsRows
        ? Math.min(teacherRowsRaw.length, logitsRows.rows)
        : Math.min(teacherRowsRaw.length, studentProbRowsFallback.length);
      if (rowCount <= 0) {
        throw new Error('Distill KD objective requires at least one aligned teacher/student row.');
      }

      const gradValues = logitsRows
        ? new Float32Array(logitsRows.rows * logitsRows.cols)
        : null;
      let kdSum = 0;
      let ceAuxSum = 0;
      for (let row = 0; row < rowCount; row += 1) {
        const teacherTokens = Array.isArray(teacherTokenRows[row]) ? teacherTokenRows[row] : [];
        const expectedSize = teacherRowsRaw[row]?.length || 1;
        const studentProbs = logitsRows
          ? softmax(
            gatherStudentLogitsRow(logitsRows.slices[row], teacherTokens, expectedSize),
            temperature
          )
          : normalizeProbRow(studentProbRowsFallback[row], teacherRowsRaw[row]?.length || 1);
        const teacherProbs = normalizeProbRow(teacherRowsRaw[row], studentProbs.length);
        kdSum += klDivergence(teacherProbs, studentProbs);
        let targetIndex = Number.isInteger(teacherTargets[row])
          ? teacherTargets[row]
          : argmax(teacherProbs);
        const targetTokenId = Number.isInteger(teacherTargetTokenIds[row])
          ? teacherTargetTokenIds[row]
          : null;
        if (targetTokenId !== null && teacherTokens.length > 0) {
          const tokenPosition = teacherTokens.indexOf(targetTokenId);
          if (tokenPosition >= 0) {
            targetIndex = tokenPosition;
          }
        }
        const clampedTarget = Math.max(0, Math.min(studentProbs.length - 1, targetIndex));
        ceAuxSum += -Math.log(Math.max(EPS, studentProbs[clampedTarget]));

        if (logitsRows && gradValues) {
          const rowOffset = row * logitsRows.cols;
          for (let col = 0; col < studentProbs.length; col += 1) {
            const mappedToken = Number(teacherTokens[col]);
            const mappedCol = Number.isInteger(mappedToken) && mappedToken >= 0 && mappedToken < logitsRows.cols
              ? mappedToken
              : (col < logitsRows.cols ? col : -1);
            if (mappedCol < 0) continue;
            const studentProb = studentProbs[col];
            const teacherProb = col < teacherProbs.length ? teacherProbs[col] : 0;
            const targetOneHot = col === clampedTarget ? 1 : 0;
            const grad = ((alphaKd * (studentProb - teacherProb)) + (alphaCe * (studentProb - targetOneHot))) / temperature;
            gradValues[rowOffset + mappedCol] += grad / rowCount;
          }
        }
      }

      const lossKd = alphaKd * (kdSum / rowCount);
      const lossCe = alphaCe * (ceAuxSum / rowCount);
      const teacherModelId = batch?.distill?.teacherModelId || distill.teacherModelId || null;
      const studentModelId = batch?.distill?.studentModelId || distill.studentModelId || null;
      const logitsDtype = forwardState.logits?.dtype;
      if (logitsRows && gradValues && logitsDtype === undefined) {
        throw new Error('distill_kd: forwardState.logits.dtype is required for backward pass');
      }
      return {
        loss,
        components: {
          loss_kd: lossKd,
          distill_stage: 'stage_a',
          distill_temperature: temperature,
          distill_alpha_kd: alphaKd,
          distill_alpha_ce: alphaCe,
          distill_loss_ce_aux: lossCe,
          distill_loss_total: lossKd + lossCe,
          distill_teacher_model_id: teacherModelId,
          distill_student_model_id: studentModelId,
        },
        _distillBackward: logitsRows && gradValues
          ? {
            logitsShape: [logitsRows.rows, logitsRows.cols],
            logitsDtype: logitsDtype,
            logitsGradValues: gradValues,
          }
          : null,
      };
    },
    backwardTargets({ loss, lossScale, lossResult, forwardState }) {
      const seeds = [];
      const ceSeed = createLossGradient(loss, lossScale);
      seeds.push({ tensor: loss, grad: ceSeed });
      const distillBackward = lossResult?._distillBackward;
      if (
        distillBackward
        && Array.isArray(distillBackward.logitsShape)
        && (distillBackward.logitsGradValues instanceof Float32Array)
        && isTensorLike(forwardState?.logits)
      ) {
        const kdSeed = createGradientTensor(
          distillBackward.logitsGradValues,
          distillBackward.logitsShape,
          distillBackward.logitsDtype,
          'distill_kd_logits_grad_output'
        );
        seeds.push({ tensor: forwardState.logits, grad: kdSeed });
      }
      return { seeds };
    },
    metrics({ config, lossResult }) {
      const distill = resolveDistillConfig(config);
      const components = lossResult.components || {};
      return {
        loss_kd: Number.isFinite(components.loss_kd) ? components.loss_kd : 0,
        distill_stage: 'stage_a',
        distill_temperature: Number.isFinite(components.distill_temperature)
          ? components.distill_temperature
          : toFinite(distill.temperature, 1),
        distill_alpha_kd: Number.isFinite(components.distill_alpha_kd)
          ? components.distill_alpha_kd
          : toFinite(distill.alphaKd, 1),
        distill_alpha_ce: Number.isFinite(components.distill_alpha_ce)
          ? components.distill_alpha_ce
          : toFinite(distill.alphaCe, 0),
        distill_loss_ce_aux: Number.isFinite(components.distill_loss_ce_aux)
          ? components.distill_loss_ce_aux
          : 0,
        distill_loss_total: Number.isFinite(components.distill_loss_total)
          ? components.distill_loss_total
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
