#!/usr/bin/env node

import process from 'node:process';

import { loadBackwardRegistry } from '../src/config/backward-registry-loader.js';
import { AutogradTape } from '../src/experimental/training/autograd.js';
import {
  createDistillStudentRuntimeModelFixture,
  loadDistillModelHandle,
} from '../src/experimental/training/suite.js';
import { releaseBuffer, readBuffer } from '../src/memory/buffer-pool.js';
import { installNodeFileFetchShim } from '../src/tooling/node-file-fetch.js';
import { bootstrapNodeWebGPU } from '../src/tooling/node-webgpu.js';

const DEFAULT_MODEL = 'gemma-3-270m-it-f16-af32';
const DEFAULT_PROMPT = 'Write a JavaScript function that adds two finite numbers.';

function printHelp() {
  console.log(`Usage: node tools/inspect-student-forward-parity.js [options]

Compare production prefill logits with the trainable full-transformer student graph.

Options:
  --model <ref>       Local model id, path, or URL (default: ${DEFAULT_MODEL})
  --prompt <text>     Prompt rendered without a chat template
  --max-abs <value>   Maximum allowed absolute logit delta (default: 0.25)
  --mean-abs <value>  Maximum allowed mean absolute logit delta (default: 0.02)
  --diagnose-layer N  Compare matching semantic stages for layer N
  --json              Print the receipt as JSON
  --help              Show this help
`);
}

function requirePositiveFinite(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive finite number.`);
  }
  return parsed;
}

function parseArgs(argv) {
  const options = {
    model: DEFAULT_MODEL,
    prompt: DEFAULT_PROMPT,
    maxAbs: 0.25,
    meanAbs: 0.02,
    diagnoseLayer: null,
    json: false,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--model') {
      options.model = String(argv[index + 1] || '').trim();
      index += 1;
      continue;
    }
    if (arg === '--prompt') {
      options.prompt = String(argv[index + 1] || '').trim();
      index += 1;
      continue;
    }
    if (arg === '--max-abs') {
      options.maxAbs = requirePositiveFinite(argv[index + 1], '--max-abs');
      index += 1;
      continue;
    }
    if (arg === '--mean-abs') {
      options.meanAbs = requirePositiveFinite(argv[index + 1], '--mean-abs');
      index += 1;
      continue;
    }
    if (arg === '--diagnose-layer') {
      const layer = Number(argv[index + 1]);
      if (!Number.isInteger(layer) || layer < 0) {
        throw new Error('--diagnose-layer must be a non-negative integer.');
      }
      options.diagnoseLayer = layer;
      index += 1;
      continue;
    }
    if (arg === '--json') {
      options.json = true;
      continue;
    }
    if (arg === '--help') {
      options.help = true;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!options.model) {
    throw new Error('--model must be non-empty.');
  }
  if (!options.prompt) {
    throw new Error('--prompt must be non-empty.');
  }
  return options;
}

function argmax(values) {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value > bestValue) {
      bestIndex = index;
      bestValue = value;
    }
  }
  return { index: bestIndex, value: bestValue };
}

function collectProtectedBuffers(model) {
  const protectedBuffers = new Set();
  const groups = typeof model?.paramGroups === 'function' ? model.paramGroups() : {};
  for (const tensors of Object.values(groups || {})) {
    for (const tensor of Array.isArray(tensors) ? tensors : []) {
      if (tensor?.buffer) protectedBuffers.add(tensor.buffer);
    }
  }
  return protectedBuffers;
}

function disposeTapeOutputs(tape, protectedBuffers) {
  const released = new Set();
  for (const record of tape?.records || []) {
    const buffer = record?.output?.buffer;
    if (!buffer || protectedBuffers.has(buffer) || released.has(buffer)) continue;
    released.add(buffer);
    releaseBuffer(buffer);
  }
}

function compareLogits(reference, candidate) {
  if (reference.length !== candidate.length || reference.length === 0) {
    throw new Error(
      `Logit length mismatch: production=${reference.length}, training=${candidate.length}.`
    );
  }
  let maxAbsDelta = 0;
  let sumAbsDelta = 0;
  let dot = 0;
  let referenceSq = 0;
  let candidateSq = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const referenceValue = reference[index];
    const candidateValue = candidate[index];
    const absDelta = Math.abs(referenceValue - candidateValue);
    maxAbsDelta = Math.max(maxAbsDelta, absDelta);
    sumAbsDelta += absDelta;
    dot += referenceValue * candidateValue;
    referenceSq += referenceValue * referenceValue;
    candidateSq += candidateValue * candidateValue;
  }
  return {
    maxAbsDelta,
    meanAbsDelta: sumAbsDelta / reference.length,
    cosineSimilarity: dot / Math.sqrt(referenceSq * candidateSq),
    productionTop1: argmax(reference),
    trainingTop1: argmax(candidate),
  };
}

function stageKey(stage, layerIdx) {
  return layerIdx == null ? stage : `layer.${layerIdx}.${stage}`;
}

async function readTensorValues(tensor) {
  const elements = tensor.shape.reduce((product, value) => product * value, 1);
  const bytesPerElement = tensor.dtype === 'f16' ? 2 : 4;
  const raw = await readBuffer(tensor.buffer, elements * bytesPerElement);
  if (tensor.dtype === 'f16') {
    throw new Error('Training parity stage capture currently requires f32 tensors.');
  }
  return Array.from(new Float32Array(raw));
}

function compareStageVectors(reference, candidate) {
  if (reference.length !== candidate.length || reference.length === 0) {
    return {
      comparable: false,
      productionElements: reference.length,
      trainingElements: candidate.length,
    };
  }
  const comparison = compareLogits(reference, candidate);
  let maxAbsIndex = 0;
  let maxAbsDelta = -1;
  for (let index = 0; index < reference.length; index += 1) {
    const delta = Math.abs(reference[index] - candidate[index]);
    if (delta > maxAbsDelta) {
      maxAbsDelta = delta;
      maxAbsIndex = index;
    }
  }
  return {
    comparable: true,
    maxAbsDelta: comparison.maxAbsDelta,
    meanAbsDelta: comparison.meanAbsDelta,
    cosineSimilarity: comparison.cosineSimilarity,
    maxAbsIndex,
    productionAtMax: reference[maxAbsIndex],
    trainingAtMax: candidate[maxAbsIndex],
    productionSample: reference.slice(0, 8),
    trainingSample: candidate.slice(0, 8),
  };
}

function collectProductionDiagnostics(pipeline) {
  const timeline = pipeline.getStats?.()?.operatorDiagnostics?.timeline || [];
  const stages = new Map();
  for (const record of timeline) {
    if (!Array.isArray(record?.capture?.data)) continue;
    stages.set(record.opId, record.capture.data);
  }
  return stages;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  installNodeFileFetchShim();
  const gpu = await bootstrapNodeWebGPU();
  if (!gpu?.ok) {
    throw new Error(`Node WebGPU bootstrap failed: ${gpu?.detail || 'unknown error'}.`);
  }

  const handle = await loadDistillModelHandle(options.model, 'student parity', {
    runtime: {
      runtimeConfig: {
        shared: {
          debug: {
            pipeline: {
              enabled: true,
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
    },
  });
  let fixture = null;
  let reference = null;
  let tape = null;
  let productionStages = new Map();
  try {
    fixture = await createDistillStudentRuntimeModelFixture({}, {
      distillRuntime: {
        studentPipeline: handle.pipeline,
        studentGraphMode: 'transformer_full',
      },
      studentGraphMode: 'transformer_full',
    });
    handle.pipeline.reset();
    reference = await handle.pipeline.prefillWithLogits(options.prompt, {
      useChatTemplate: false,
      diagnostics: options.diagnoseLayer == null
        ? null
        : {
          enabled: true,
          captureConfig: {
            enabled: true,
            defaultLevel: 'metadata',
            targetLevel: 'full',
            targetOpIds: ['embed.out', 'final_norm.pre', 'final_norm.out', 'logits.out'],
            targetLayers: [options.diagnoseLayer],
          },
        },
    });
    productionStages = collectProductionDiagnostics(handle.pipeline);
    const productionTokens = Array.from(reference.tokens || []);
    const trainingTokens = Array.from(handle.pipeline.tokenizer.encode(options.prompt));
    if (
      productionTokens.length !== trainingTokens.length
      || productionTokens.some((token, index) => token !== trainingTokens[index])
    ) {
      throw new Error(
        `Tokenizer input mismatch: production=${productionTokens.length}, training=${trainingTokens.length}.`
      );
    }

    handle.pipeline.reset();
    tape = new AutogradTape(loadBackwardRegistry());
    const trainingStages = new Map();
    const result = await fixture.model.forwardDistill({
      distill: { prompts: [options.prompt] },
    }, tape, {
      captureStage: options.diagnoseLayer == null
        ? null
        : async ({ stage, layerIdx, tensor }) => {
          if (layerIdx != null && layerIdx !== options.diagnoseLayer) return;
          trainingStages.set(stageKey(stage, layerIdx), await readTensorValues(tensor));
        },
    });
    const trainingLogits = new Float32Array(await readBuffer(result.logits.buffer));
    const productionLogits = reference.logits instanceof Float32Array
      ? reference.logits
      : Float32Array.from(reference.logits || []);
    const comparison = compareLogits(productionLogits, trainingLogits);
    const passed = comparison.maxAbsDelta <= options.maxAbs
      && comparison.meanAbsDelta <= options.meanAbs
      && comparison.productionTop1.index === comparison.trainingTop1.index;
    let stageDiagnostics = null;
    if (options.diagnoseLayer != null) {
      stageDiagnostics = [];
      for (const [key, trainingValues] of trainingStages) {
        const productionValues = productionStages.get(key);
        if (!productionValues) continue;
        stageDiagnostics.push({
          stage: key,
          ...compareStageVectors(productionValues, trainingValues),
        });
      }
    }
    const receipt = {
      artifactType: 'student_forward_parity_receipt',
      schemaVersion: 1,
      modelId: handle.manifest?.modelId || options.model,
      prompt: options.prompt,
      tokenCount: productionTokens.length,
      vocabSize: productionLogits.length,
      thresholds: {
        maxAbsDelta: options.maxAbs,
        meanAbsDelta: options.meanAbs,
        requireTop1Match: true,
      },
      ...comparison,
      stageDiagnostics,
      passed,
    };
    if (options.json) {
      console.log(JSON.stringify(receipt, null, 2));
    } else {
      console.log(
        `student-forward-parity: ${passed ? 'PASS' : 'FAIL'} `
        + `max_abs=${comparison.maxAbsDelta} mean_abs=${comparison.meanAbsDelta} `
        + `cosine=${comparison.cosineSimilarity} `
        + `top1=${comparison.productionTop1.index}/${comparison.trainingTop1.index}`
      );
    }
    if (!passed) process.exitCode = 1;
  } finally {
    reference?.cache?.clear?.();
    fixture?.model?.cleanupDistillStep?.();
    if (fixture && tape) {
      disposeTapeOutputs(tape, collectProtectedBuffers(fixture.model));
    }
    fixture?.cleanup?.();
    handle.pipeline.reset();
    await handle.pipeline.unload?.();
  }
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
