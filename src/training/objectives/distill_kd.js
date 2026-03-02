import { crossEntropyLoss as defaultCrossEntropyLoss } from '../loss.js';
import { createTrainingObjective } from './base.js';
import { readBuffer } from '../../memory/buffer-pool.js';
import { f16ToF32Array } from '../../inference/kv-cache/types.js';

const EPS = 1e-8;

function toFinite(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function resolveDistillConfig(config) {
  return config?.training?.distill || {};
}

async function readLogitsRows(logitsTensor) {
  if (!logitsTensor?.buffer || !Array.isArray(logitsTensor?.shape) || logitsTensor.shape.length < 2) {
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
  return slices;
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

export function createDistillKdObjective(options = {}) {
  const lossFn = options.crossEntropyLoss || defaultCrossEntropyLoss;
  if (typeof lossFn !== 'function') {
    throw new Error('Distill KD objective requires crossEntropyLoss(logits, targets, config, tape).');
  }

  return createTrainingObjective({
    name: 'kd',
    async forward({ model, batch, tape }) {
      const logits = await model.forward(batch.input, tape);
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

      let studentRows = null;
      try {
        const logitsRows = await readLogitsRows(forwardState.logits);
        studentRows = logitsRows.map((row) => softmax(row, temperature));
      } catch {
        studentRows = null;
      }
      if (!studentRows) {
        if (Array.isArray(batch?.distill?.studentTopProbs) && batch.distill.studentTopProbs.length > 0) {
          studentRows = batch.distill.studentTopProbs.map((row) => normalizeProbRow(row, teacherRowsRaw[0].length));
        } else if (Array.isArray(batch?.distill?.studentTopLogits) && batch.distill.studentTopLogits.length > 0) {
          studentRows = batch.distill.studentTopLogits.map((row) => {
            const logits = new Float32Array(teacherRowsRaw[0].length);
            if (Array.isArray(row) || ArrayBuffer.isView(row)) {
              const source = Array.isArray(row) ? row : Array.from(row);
              const count = Math.min(logits.length, source.length);
              for (let i = 0; i < count; i += 1) {
                logits[i] = toFinite(source[i], 0);
              }
            }
            return softmax(logits, temperature);
          });
        } else {
          throw new Error(
            'Distill KD objective requires forward logits readback or batch.distill.studentTopProbs.'
          );
        }
      }

      const teacherTargets = Array.isArray(batch?.distill?.teacherTargetIndices)
        ? batch.distill.teacherTargetIndices
        : [];
      let kdSum = 0;
      let ceAuxSum = 0;
      let rowCount = 0;
      const rowLimit = Math.min(teacherRowsRaw.length, studentRows.length);
      for (let row = 0; row < rowLimit; row += 1) {
        const studentProbs = normalizeProbRow(studentRows[row], studentRows[row].length);
        const teacherProbs = normalizeProbRow(teacherRowsRaw[row], studentProbs.length);
        kdSum += klDivergence(teacherProbs, studentProbs);
        const targetIndex = Number.isInteger(teacherTargets[row])
          ? teacherTargets[row]
          : argmax(teacherProbs);
        const clampedTarget = Math.max(0, Math.min(studentProbs.length - 1, targetIndex));
        ceAuxSum += -Math.log(Math.max(EPS, studentProbs[clampedTarget]));
        rowCount += 1;
      }
      const lossKd = rowCount > 0 ? (alphaKd * (kdSum / rowCount)) : 0;
      const lossCe = rowCount > 0 ? (alphaCe * (ceAuxSum / rowCount)) : 0;
      const teacherModelId = batch?.distill?.teacherModelId || distill.teacherModelId || null;
      const studentModelId = batch?.distill?.studentModelId || distill.studentModelId || null;
      return {
        loss,
        components: {
          loss_kd: lossKd,
          distill_stage: 'stage_a',
          distill_temperature: temperature,
          distill_alpha_kd: alphaKd,
          distill_alpha_ce: alphaCe,
          distill_loss_ce_aux: lossCe,
          distill_teacher_model_id: teacherModelId,
          distill_student_model_id: studentModelId,
        },
      };
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
        distill_teacher_model_id: typeof components.distill_teacher_model_id === 'string'
          ? components.distill_teacher_model_id
          : (distill.teacherModelId || null),
        distill_student_model_id: typeof components.distill_student_model_id === 'string'
          ? components.distill_student_model_id
          : (distill.studentModelId || null),
      };
    },
  });
}
