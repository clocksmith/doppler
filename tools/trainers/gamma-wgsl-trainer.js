import { execFile } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';

import { parseSafetensorsHeader } from '../../src/formats/safetensors/types.js';

const execFileAsync = promisify(execFile);
const PROTOCOL = 'gamma_wgsl_trainer_json_v1';
const TARGET_MODULES = new Set([
  'q_proj',
  'k_proj',
  'v_proj',
  'o_proj',
  'gate_proj',
  'up_proj',
  'down_proj',
]);

function float16ToFloat32(value) {
  const sign = (value & 0x8000) << 16;
  let exponent = (value >>> 10) & 0x1f;
  let fraction = value & 0x03ff;
  if (exponent === 0) {
    if (fraction === 0) {
      const bits = sign;
      const view = new DataView(new ArrayBuffer(4));
      view.setUint32(0, bits, true);
      return view.getFloat32(0, true);
    }
    while ((fraction & 0x0400) === 0) {
      fraction <<= 1;
      exponent -= 1;
    }
    exponent += 1;
    fraction &= ~0x0400;
  } else if (exponent === 31) {
    exponent = 255;
  }
  exponent = exponent + (127 - 15);
  const bits = sign | (exponent << 23) | (fraction << 13);
  const view = new DataView(new ArrayBuffer(4));
  view.setUint32(0, bits, true);
  return view.getFloat32(0, true);
}

function readTensorValues(bytes, tensor) {
  const elements = tensor.shape.reduce((product, value) => product * Number(value), 1);
  const values = new Float32Array(elements);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < elements; index += 1) {
    if (tensor.dtypeOriginal === 'F32') {
      values[index] = view.getFloat32(tensor.offset + (index * 4), true);
    } else if (tensor.dtypeOriginal === 'F16') {
      values[index] = float16ToFloat32(view.getUint16(tensor.offset + (index * 2), true));
    } else if (tensor.dtypeOriginal === 'BF16') {
      const bits = view.getUint16(tensor.offset + (index * 2), true) << 16;
      const scratch = new DataView(new ArrayBuffer(4));
      scratch.setUint32(0, bits, true);
      values[index] = scratch.getFloat32(0, true);
    } else {
      throw new Error(`Unsupported Gamma adapter dtype: ${tensor.dtypeOriginal}`);
    }
  }
  return values;
}

function transpose(values, rows, columns) {
  const output = new Float32Array(values.length);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      output[(column * rows) + row] = values[(row * columns) + column];
    }
  }
  return output;
}

function normalizePeftTensor(tensor, values) {
  const match = tensor.name.match(
    /(?:^|\.)layers\.(\d+)\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_([AB])(?:\.[^.]+)?\.weight$/
  );
  if (!match || !TARGET_MODULES.has(match[2])) return null;
  if (tensor.shape.length !== 2) {
    throw new Error(`Gamma adapter tensor ${tensor.name} must be rank 2.`);
  }
  const [rows, columns] = tensor.shape.map(Number);
  const kind = match[3] === 'A' ? 'a' : 'b';
  return {
    name: `layers.${match[1]}.${match[2]}.lora_${kind}`,
    shape: [columns, rows],
    dtype: 'f32',
    tensor: transpose(values, rows, columns),
  };
}

export async function readGammaAdapterTensors(adapterPath) {
  const weightsPath = join(resolve(adapterPath), 'adapter_model.safetensors');
  const bytes = await readFile(weightsPath);
  const arrayBuffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const parsed = parseSafetensorsHeader(arrayBuffer);
  const tensors = [];
  for (const tensor of parsed.tensors) {
    const normalized = normalizePeftTensor(tensor, readTensorValues(bytes, tensor));
    if (normalized) tensors.push(normalized);
  }
  if (tensors.length === 0) {
    throw new Error(`Gamma adapter ${weightsPath} contains no supported LoRA tensors.`);
  }
  return tensors;
}

function gammaPaths() {
  const gammaRoot = resolve(
    process.env.GAMMA_ROOT || join(process.cwd(), '..', 'gamma')
  );
  return {
    gammaRoot,
    python: resolve(process.env.GAMMA_WGSL_PYTHON || join(gammaRoot, '.venv_rocm', 'bin', 'python')),
    trainer: join(
      gammaRoot,
      'projects',
      'distillation',
      'wgsl',
      'training',
      'train_wgsl.py'
    ),
  };
}

export function buildGammaProcessEnv(environment = process.env) {
  return {
    ...environment,
    ...(environment.GAMMA_WGSL_PYTHONPATH
      ? { PYTHONPATH: environment.GAMMA_WGSL_PYTHONPATH }
      : {}),
  };
}

export async function runGammaWgslRequest(request, options = {}) {
  if (request?.protocol !== PROTOCOL) {
    throw new Error(`Gamma WGSL request.protocol must be ${PROTOCOL}.`);
  }
  const paths = gammaPaths();
  const runRoot = resolve(options.runRoot || request.outputRoot);
  await mkdir(runRoot, { recursive: true });
  const prefix = String(options.prefix || request.action || 'request');
  const requestPath = join(runRoot, `${prefix}-request.json`);
  const responsePath = join(runRoot, `${prefix}-response.json`);
  const stdoutPath = join(runRoot, `${prefix}-stdout.log`);
  const stderrPath = join(runRoot, `${prefix}-stderr.log`);
  await writeFile(requestPath, `${JSON.stringify(request, null, 2)}\n`, 'utf8');
  let processResult;
  try {
    processResult = await execFileAsync(
      paths.python,
      [paths.trainer, '--request', requestPath, '--response', responsePath],
      {
        cwd: paths.gammaRoot,
        env: buildGammaProcessEnv(),
        encoding: 'utf8',
        maxBuffer: 16 * 1024 * 1024,
      }
    );
  } catch (cause) {
    await Promise.all([
      writeFile(stdoutPath, String(cause?.stdout || ''), 'utf8'),
      writeFile(stderrPath, String(cause?.stderr || cause?.message || cause), 'utf8'),
    ]);
    let response = null;
    try {
      response = JSON.parse(await readFile(responsePath, 'utf8'));
    } catch {
      // Raw process logs are the evidence when the trainer cannot write JSON.
    }
    const error = new Error(`gamma_wgsl_trainer_failed: ${response?.error || cause?.message || cause}`);
    error.response = response;
    error.paths = { requestPath, responsePath, stdoutPath, stderrPath };
    throw error;
  }
  await Promise.all([
    writeFile(stdoutPath, processResult.stdout || '', 'utf8'),
    writeFile(stderrPath, processResult.stderr || '', 'utf8'),
  ]);
  const response = JSON.parse(await readFile(responsePath, 'utf8'));
  if (response.protocol !== PROTOCOL || response.ok !== true || response.action !== request.action) {
    throw new Error(`gamma_wgsl_trainer_invalid_response: ${response.error || 'contract mismatch'}`);
  }
  return {
    response,
    paths: { requestPath, responsePath, stdoutPath, stderrPath },
  };
}

function precisionName(value) {
  if (value === 'bf16' || value === 'bfloat16') return 'bfloat16';
  if (value === 'f16' || value === 'float16') return 'float16';
  if (value === 'f32' || value === 'float32') return 'float32';
  throw new Error(`Unsupported Gamma WGSL precision: ${value}`);
}

export function buildGammaRequest(input, outputRoot) {
  const pipeline = input.workload.pipeline;
  const modelId = String(pipeline.baseModelRef || input.workload.baseModelId);
  const localPath = String(process.env.GAMMA_WGSL_MODEL_PATH || '').trim();
  return {
    protocol: PROTOCOL,
    action: 'sft',
    runId: input.workload.id,
    outputRoot,
    model: {
      modelId,
      revision: pipeline.baseModelRevision || process.env.GAMMA_WGSL_MODEL_REVISION || 'main',
      ...(localPath ? { localPath } : {}),
    },
    adapter: {
      rank: input.adapter.rank,
      alpha: input.adapter.alpha,
      dropout: input.adapter.dropout,
      targetModules: input.adapter.targetModules,
    },
    datasetPath: input.dataset.absolutePath,
    training: {
      dtype: precisionName(input.training.precision.activations),
      gradientCheckpointing: true,
      steps: input.training.steps,
      gradientAccumulationSteps: input.training.accumSteps,
      maxLength: pipeline.maxLength || pipeline.sequenceLength,
      learningRate: input.training.optimizer.lr,
      weightDecay: input.training.optimizer.weightDecay,
      maxGradNorm: input.training.gradientClipping.maxNorm,
      seed: input.workload.seed,
      ...(pipeline.rowOrder ? { rowOrder: pipeline.rowOrder } : {}),
    },
  };
}

export async function trainCausalLmLora(input) {
  const paths = gammaPaths();
  const gammaRoot = join(input.layout.checkpoints, 'gamma');
  const requestPath = join(input.layout.logs, 'gamma-request.json');
  const responsePath = join(input.layout.logs, 'gamma-response.json');
  const stdoutPath = join(input.layout.logs, 'gamma-stdout.log');
  const stderrPath = join(input.layout.logs, 'gamma-stderr.log');
  await mkdir(dirname(requestPath), { recursive: true });
  const request = buildGammaRequest(input, gammaRoot);
  await writeFile(requestPath, `${JSON.stringify(request, null, 2)}\n`, 'utf8');
  let processResult;
  try {
    processResult = await execFileAsync(
      paths.python,
      [paths.trainer, '--request', requestPath, '--response', responsePath],
      {
        cwd: paths.gammaRoot,
        env: buildGammaProcessEnv(),
        encoding: 'utf8',
        maxBuffer: 16 * 1024 * 1024,
      }
    );
  } catch (error) {
    await Promise.all([
      writeFile(stdoutPath, String(error?.stdout || ''), 'utf8'),
      writeFile(stderrPath, String(error?.stderr || error?.message || error), 'utf8'),
    ]);
    let response = null;
    try {
      response = JSON.parse(await readFile(responsePath, 'utf8'));
    } catch {
      // The process-level error below retains the raw logs when no JSON response exists.
    }
    throw new Error(`gamma_wgsl_trainer_failed: ${response?.error || error?.message || error}`);
  }
  await Promise.all([
    writeFile(stdoutPath, processResult.stdout || '', 'utf8'),
    writeFile(stderrPath, processResult.stderr || '', 'utf8'),
  ]);
  const response = JSON.parse(await readFile(responsePath, 'utf8'));
  if (response.protocol !== PROTOCOL || response.ok !== true || response.action !== 'sft') {
    throw new Error(`gamma_wgsl_trainer_invalid_response: ${response.error || 'contract mismatch'}`);
  }
  const result = response.result;
  const tensors = await readGammaAdapterTensors(result.adapterPath);
  return {
    checkpointId: `checkpoint-${String(result.checkpointStep).padStart(6, '0')}`,
    checkpointStep: result.checkpointStep,
    trainerId: 'gamma-wgsl-rocm-v1',
    runnerId: 'gamma-wgsl-causal-lm-lora',
    metrics: result.metrics,
    receipts: [{
      protocol: response.protocol,
      requestHash: response.requestHash,
      runtime: response.runtime,
      policyHash: result.policyHash,
      modelPath: result.modelPath,
      adapterPath: result.adapterPath,
      metricsPath: result.metricsPath,
      requestPath,
      responsePath,
      stdoutPath,
      stderrPath,
      claimBoundary: response.claimBoundary,
    }],
    tensors,
  };
}

trainCausalLmLora.runnerId = 'gamma-wgsl-causal-lm-lora';
