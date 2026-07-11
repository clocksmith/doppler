#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';

import { computeSampleStats } from '../src/debug/stats.js';
import {
  getRuntimeConfig,
  resetRuntimeConfig,
  setRuntimeConfig,
} from '../src/config/runtime.js';
import {
  computeEvalMetrics,
  computeExactMatch,
} from '../src/experimental/training/operator-eval.js';
import { parseJsonl } from '../src/experimental/training/datasets/jsonl.js';
import { applyRuntimeProfile } from '../src/inference/browser-harness-runtime-helpers.js';
import { applyRuntimeInputs } from '../src/tooling/command-runner-shared.js';
import { runNodeCommand } from '../src/tooling/node-command-runner.js';

const SACREBLEU_PROGRAM = [
  'import json, sys',
  'import sacrebleu',
  'payload = json.load(sys.stdin)',
  'results = {}',
  'for group_id, group in payload["groups"].items():',
  '    bleu_metric = sacrebleu.metrics.BLEU()',
  '    chrf_metric = sacrebleu.metrics.CHRF()',
  '    bleu = bleu_metric.corpus_score(group["hypotheses"], [group["references"]])',
  '    chrf = chrf_metric.corpus_score(group["hypotheses"], [group["references"]])',
  '    results[group_id] = {',
  '        "bleu": float(bleu.score),',
  '        "bleuSignature": str(bleu_metric.get_signature()),',
  '        "chrf": float(chrf.score),',
  '        "chrfSignature": str(chrf_metric.get_signature()),',
  '        "count": len(group["hypotheses"]),',
  '    }',
  'json.dump({"sacrebleuVersion": sacrebleu.__version__, "groups": results}, sys.stdout, sort_keys=True)',
].join('\n');

function parseArgs(argv) {
  let configPath = null;
  let help = false;
  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--help' || arg === '-h') {
      help = true;
      continue;
    }
    if (arg === '--config') {
      const value = argv[index + 1];
      if (!value || value.startsWith('--')) {
        throw new Error('--config requires a JSON file path.');
      }
      configPath = value;
      index += 1;
      continue;
    }
    throw new Error(`Unknown flag: ${arg}`);
  }
  return { configPath, help };
}

function usage() {
  return [
    'Usage:',
    '  node tools/bench-translation-quality.js --config <path>',
    '',
    'The config must define separate sharedContract and engineLanes blocks.',
  ].join('\n');
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function requireObject(value, label) {
  if (!isPlainObject(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function requireString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value.trim();
}

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return value;
}

function sha256(value) {
  return `sha256:${createHash('sha256').update(value).digest('hex')}`;
}

function compactTimestamp(date) {
  return date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/u, 'Z');
}

function resolveConfigPath(configDir, value, label) {
  return resolve(configDir, requireString(value, label));
}

function validateSampling(value) {
  const sampling = requireObject(value, 'sharedContract.sampling');
  const required = ['temperature', 'topP', 'topK', 'repetitionPenalty', 'greedyThreshold', 'seed'];
  for (const key of required) {
    if (!Number.isFinite(sampling[key])) {
      throw new Error(`sharedContract.sampling.${key} must be finite.`);
    }
  }
  return sampling;
}

function validateEngineLane(value, index) {
  const lane = requireObject(value, `engineLanes[${index}]`);
  const runtimeConfig = requireObject(lane.runtimeConfig, `engineLanes[${index}].runtimeConfig`);
  const inference = requireObject(runtimeConfig.inference, `engineLanes[${index}].runtimeConfig.inference`);
  for (const forbidden of ['prompt', 'generation', 'sampling']) {
    if (inference[forbidden] !== undefined) {
      throw new Error(
        `engineLanes[${index}].runtimeConfig.inference.${forbidden} belongs in sharedContract.`
      );
    }
  }
  return {
    id: requireString(lane.id, `engineLanes[${index}].id`),
    label: requireString(lane.label, `engineLanes[${index}].label`),
    runtimeConfig,
  };
}

async function loadConfig(configPath) {
  const absoluteConfigPath = resolve(configPath);
  const configDir = dirname(absoluteConfigPath);
  const raw = await readFile(absoluteConfigPath, 'utf8');
  const config = JSON.parse(raw);
  requireObject(config, 'config');
  if (config.schemaVersion !== 1) {
    throw new Error('config.schemaVersion must be 1.');
  }
  const model = requireObject(config.model, 'model');
  const sharedContract = requireObject(config.sharedContract, 'sharedContract');
  const metrics = requireObject(config.metrics, 'metrics');
  const engineLanes = Array.isArray(config.engineLanes)
    ? config.engineLanes.map(validateEngineLane)
    : null;
  if (!engineLanes || engineLanes.length < 2) {
    throw new Error('engineLanes must contain at least two lanes.');
  }
  if (new Set(engineLanes.map((lane) => lane.id)).size !== engineLanes.length) {
    throw new Error('engineLanes ids must be unique.');
  }
  const sampling = validateSampling(sharedContract.sampling);
  return {
    absoluteConfigPath,
    configHash: sha256(raw),
    suiteId: requireString(config.suiteId, 'suiteId'),
    model: {
      modelId: requireString(model.modelId, 'model.modelId'),
      modelDir: resolveConfigPath(configDir, model.modelDir, 'model.modelDir'),
      sourceCheckpointId: requireString(model.sourceCheckpointId, 'model.sourceCheckpointId'),
    },
    sharedContract: {
      datasetPath: resolveConfigPath(
        configDir,
        sharedContract.datasetPath,
        'sharedContract.datasetPath'
      ),
      baselinePredictionsPath: resolveConfigPath(
        configDir,
        sharedContract.baselinePredictionsPath,
        'sharedContract.baselinePredictionsPath'
      ),
      sampleLimit: requirePositiveInteger(
        sharedContract.sampleLimit,
        'sharedContract.sampleLimit'
      ),
      maxTokens: requirePositiveInteger(sharedContract.maxTokens, 'sharedContract.maxTokens'),
      warmupRuns: requirePositiveInteger(sharedContract.warmupRuns, 'sharedContract.warmupRuns'),
      determinismSamples: requirePositiveInteger(
        sharedContract.determinismSamples,
        'sharedContract.determinismSamples'
      ),
      useChatTemplate: sharedContract.useChatTemplate === true,
      cacheMode: requireString(sharedContract.cacheMode, 'sharedContract.cacheMode'),
      loadMode: requireString(sharedContract.loadMode, 'sharedContract.loadMode'),
      sampling: { ...sampling },
    },
    runtimeProfile: requireString(config.runtimeProfile, 'runtimeProfile'),
    engineLanes,
    metrics: {
      sacrebleuPython: resolveConfigPath(
        configDir,
        metrics.sacrebleuPython,
        'metrics.sacrebleuPython'
      ),
    },
    outputDir: resolveConfigPath(configDir, config.outputDir, 'outputDir'),
  };
}

function validateDatasetRows(rows, sampleLimit) {
  if (rows.length < sampleLimit) {
    throw new Error(`Dataset has ${rows.length} rows, fewer than sampleLimit=${sampleLimit}.`);
  }
  return rows.slice(0, sampleLimit).map((row, index) => {
    requireObject(row, `dataset row ${index}`);
    return {
      index,
      pair: requireString(row.pair, `dataset row ${index}.pair`),
      srcLang: requireString(row.src_lang, `dataset row ${index}.src_lang`),
      tgtLang: requireString(row.tgt_lang, `dataset row ${index}.tgt_lang`),
      source: requireString(row.source, `dataset row ${index}.source`),
      reference: requireString(row.target_pos, `dataset row ${index}.target_pos`),
    };
  });
}

function validateBaselinePredictions(rows, datasetRows) {
  if (rows.length !== datasetRows.length) {
    throw new Error(
      `Baseline predictions contain ${rows.length} rows; expected ${datasetRows.length}.`
    );
  }
  return rows.map((row, index) => {
    requireObject(row, `baseline prediction ${index}`);
    const source = requireString(row.source, `baseline prediction ${index}.source`);
    const pair = requireString(row.pair, `baseline prediction ${index}.pair`);
    if (source !== datasetRows[index].source || pair !== datasetRows[index].pair) {
      throw new Error(`Baseline prediction ${index} does not align with the dataset.`);
    }
    return requireString(row.pred, `baseline prediction ${index}.pred`);
  });
}

function composeRuntimeConfig(sharedContract, lane, prompt) {
  const runtimeConfig = structuredClone(lane.runtimeConfig);
  runtimeConfig.inference = {
    ...runtimeConfig.inference,
    prompt,
    generation: {
      maxTokens: sharedContract.maxTokens,
    },
    sampling: {
      ...sharedContract.sampling,
    },
  };
  return runtimeConfig;
}

function generationOptions(sharedContract) {
  return {
    maxTokens: sharedContract.maxTokens,
    temperature: sharedContract.sampling.temperature,
    topP: sharedContract.sampling.topP,
    topK: sharedContract.sampling.topK,
    repetitionPenalty: sharedContract.sampling.repetitionPenalty,
    greedyThreshold: sharedContract.sampling.greedyThreshold,
    seed: sharedContract.sampling.seed,
    useChatTemplate: sharedContract.useChatTemplate,
    benchmark: false,
  };
}

async function generateOne(pipeline, row, sharedContract) {
  pipeline.reset();
  const tokenIds = [];
  const chunks = [];
  const start = performance.now();
  for await (const chunk of pipeline.generate(row.source, {
    ...generationOptions(sharedContract),
    onToken(tokenId) {
      tokenIds.push(tokenId);
    },
  })) {
    chunks.push(String(chunk));
  }
  const totalMs = performance.now() - start;
  const stats = pipeline.getStats();
  const outputRaw = chunks.join('');
  return {
    index: row.index,
    pair: row.pair,
    srcLang: row.srcLang,
    tgtLang: row.tgtLang,
    source: row.source,
    reference: row.reference,
    output: outputRaw.trim(),
    outputRaw,
    outputHash: sha256(outputRaw.trim()),
    tokenIds,
    tokenIdsHash: sha256(JSON.stringify(tokenIds)),
    tokensGenerated: tokenIds.length,
    stopReason: stats.stopReason ?? null,
    stopTokenId: stats.stopTokenId ?? null,
    prefillTokens: Number(stats.prefillTokens ?? 0),
    decodeTokens: Number(stats.decodeTokens ?? tokenIds.length),
    prefillMs: Number(stats.prefillTimeMs ?? 0),
    decodeMs: Number(stats.decodeTimeMs ?? 0),
    totalMs,
    decodeRecordMs: Number(stats.decodeRecordMs ?? 0),
    decodeSubmitWaitMs: Number(stats.decodeSubmitWaitMs ?? 0),
    decodeReadbackWaitMs: Number(stats.decodeReadbackWaitMs ?? 0),
  };
}

async function loadLanePipeline(config, lane, warmupRow) {
  const modelUrl = pathToFileURL(config.model.modelDir).href.replace(/\/?$/u, '/');
  const response = await runNodeCommand({
    command: 'verify',
    workload: 'inference',
    modelId: config.model.modelId,
    modelUrl,
    loadMode: config.sharedContract.loadMode,
    cacheMode: config.sharedContract.cacheMode,
    captureOutput: true,
    keepPipeline: true,
    inferenceInput: {
      prompt: warmupRow.source,
      maxTokens: config.sharedContract.maxTokens,
    },
    runtimeProfile: config.runtimeProfile,
    runtimeConfig: composeRuntimeConfig(config.sharedContract, lane, warmupRow.source),
  });
  const pipeline = response?.result?.pipeline;
  if (!pipeline) {
    throw new Error(`${lane.id}: verify did not return a retained pipeline.`);
  }
  return {
    pipeline,
    deviceInfo: response.result.deviceInfo ?? null,
    loadMetrics: response.result.metrics?.load ?? null,
    executionPlan: response.result.metrics?.executionPlan ?? null,
    warmupOutput: response.result.output ?? null,
  };
}

async function applyLaneRuntimeContext(config, lane, warmupRow) {
  await applyRuntimeInputs({
    intent: 'verify',
    runtimeProfile: config.runtimeProfile,
    runtimeConfig: composeRuntimeConfig(config.sharedContract, lane, warmupRow.source),
  }, {
    getRuntimeConfig,
    setRuntimeConfig,
    resetRuntimeConfig,
    applyRuntimeProfile,
  });
}

function summarizeLane(predictions, determinism) {
  const totalMs = predictions.map((row) => row.totalMs);
  const prefillMs = predictions.map((row) => row.prefillMs);
  const decodeMs = predictions.map((row) => row.decodeMs);
  const decodeTokensPerSec = predictions.map((row) => (
    row.decodeMs > 0 ? (row.decodeTokens / row.decodeMs) * 1000 : 0
  ));
  const prefillTokensPerSec = predictions.map((row) => (
    row.prefillMs > 0 ? (row.prefillTokens / row.prefillMs) * 1000 : 0
  ));
  return {
    rows: predictions.length,
    maxTokenStops: predictions.filter((row) => row.tokensGenerated >= 1 && row.stopReason === 'max-tokens').length,
    timingMs: {
      total: computeSampleStats(totalMs),
      prefill: computeSampleStats(prefillMs),
      decode: computeSampleStats(decodeMs),
    },
    throughput: {
      prefillTokensPerSec: computeSampleStats(prefillTokensPerSec),
      decodeTokensPerSec: computeSampleStats(decodeTokensPerSec),
    },
    determinism,
  };
}

async function runLane(config, lane, datasetRows) {
  console.error(`[quality] lane=${lane.id} loading model`);
  const loaded = await loadLanePipeline(config, lane, datasetRows[0]);
  const { pipeline } = loaded;
  const predictions = [];
  try {
    await applyLaneRuntimeContext(config, lane, datasetRows[0]);
    for (let warmupIndex = 1; warmupIndex < config.sharedContract.warmupRuns; warmupIndex += 1) {
      await generateOne(pipeline, datasetRows[0], config.sharedContract);
    }
    for (const row of datasetRows) {
      predictions.push(await generateOne(pipeline, row, config.sharedContract));
      if (predictions.length % 8 === 0 || predictions.length === datasetRows.length) {
        console.error(`[quality] lane=${lane.id} rows=${predictions.length}/${datasetRows.length}`);
      }
    }
    const deterministicRows = [];
    const count = Math.min(config.sharedContract.determinismSamples, datasetRows.length);
    for (let index = 0; index < count; index += 1) {
      const repeat = await generateOne(pipeline, datasetRows[index], config.sharedContract);
      deterministicRows.push({
        index,
        matches: repeat.tokenIdsHash === predictions[index].tokenIdsHash,
        expectedTokenIdsHash: predictions[index].tokenIdsHash,
        repeatedTokenIdsHash: repeat.tokenIdsHash,
      });
    }
    const determinism = {
      checkedRows: deterministicRows.length,
      matchingRows: deterministicRows.filter((row) => row.matches).length,
      ok: deterministicRows.every((row) => row.matches),
      rows: deterministicRows,
    };
    return {
      id: lane.id,
      label: lane.label,
      runtimeConfig: lane.runtimeConfig,
      predictions,
      summary: summarizeLane(predictions, determinism),
      deviceInfo: loaded.deviceInfo,
      loadMetrics: loaded.loadMetrics,
      executionPlan: loaded.executionPlan,
      warmupOutput: loaded.warmupOutput,
    };
  } finally {
    try {
      await pipeline.unload();
    } finally {
      resetRuntimeConfig();
    }
  }
}

function buildMetricGroups(datasetRows, baselinePredictions, lanes) {
  const groups = {};
  const sources = [
    { id: 'source-bf16', predictions: baselinePredictions },
    ...lanes.map((lane) => ({
      id: lane.id,
      predictions: lane.predictions.map((row) => row.output),
    })),
  ];
  const pairs = [...new Set(datasetRows.map((row) => row.pair))].sort();
  for (const source of sources) {
    const allIndices = datasetRows.map((_, index) => index);
    const groupIds = [['overall', allIndices]];
    for (const pair of pairs) {
      groupIds.push([
        pair,
        datasetRows.flatMap((row, index) => row.pair === pair ? [index] : []),
      ]);
    }
    for (const [groupId, indices] of groupIds) {
      groups[`${source.id}:${groupId}`] = {
        hypotheses: indices.map((index) => source.predictions[index]),
        references: indices.map((index) => datasetRows[index].reference),
      };
    }
  }
  return groups;
}

function runPythonJson(pythonPath, program, payload) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(pythonPath, ['-c', program], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    const stdout = [];
    const stderr = [];
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    child.on('error', reject);
    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(
          `Metric process exited with code ${code}: ${Buffer.concat(stderr).toString('utf8').trim()}`
        ));
        return;
      }
      try {
        resolvePromise(JSON.parse(Buffer.concat(stdout).toString('utf8')));
      } catch (error) {
        reject(new Error(`Metric process returned invalid JSON: ${error.message}`));
      }
    });
    child.stdin.end(JSON.stringify(payload));
  });
}

function buildDopplerOperatorMetrics(datasetRows, predictions) {
  const hypotheses = predictions.map((row) => row.output);
  const references = datasetRows.map((row) => row.reference);
  const metrics = computeEvalMetrics('translation', hypotheses, references);
  const exactMatch = computeExactMatch(hypotheses, references);
  return {
    bleuRatio: metrics.bleu.score,
    chrfRatio: metrics.chrf.score,
    exactMatchRatio: exactMatch.score,
    exactMatches: exactMatch.matches,
    count: exactMatch.total,
  };
}

function comparePredictions(leftId, left, rightId, right, limit = 12) {
  if (left.length !== right.length) {
    throw new Error(`${leftId}/${rightId} prediction counts differ.`);
  }
  const divergences = [];
  let exactMatches = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftOutput = String(left[index] ?? '').trim();
    const rightOutput = String(right[index] ?? '').trim();
    if (leftOutput === rightOutput) {
      exactMatches += 1;
    } else if (divergences.length < limit) {
      divergences.push({ index, left: leftOutput, right: rightOutput });
    }
  }
  return {
    leftId,
    rightId,
    total: left.length,
    exactMatches,
    exactMatchRatio: exactMatches / left.length,
    exact: exactMatches === left.length,
    divergencePreview: divergences,
  };
}

async function writeArtifacts(config, datasetRaw, baselineRaw, datasetRows, baselinePredictions, lanes, metrics) {
  await mkdir(config.outputDir, { recursive: true });
  const createdAt = new Date();
  const stamp = compactTimestamp(createdAt);
  const artifactStem = `${config.suiteId}_${stamp}`;
  const laneArtifacts = {};
  for (const lane of lanes) {
    const filename = `${artifactStem}.${lane.id}.predictions.jsonl`;
    const content = `${lane.predictions.map((row) => JSON.stringify(row)).join('\n')}\n`;
    await writeFile(resolve(config.outputDir, filename), content, 'utf8');
    laneArtifacts[lane.id] = {
      filename,
      sha256: sha256(content),
      rows: lane.predictions.length,
    };
  }
  const sourceBaseline = {
    id: 'source-bf16',
    path: config.sharedContract.baselinePredictionsPath,
    sha256: sha256(baselineRaw),
    predictions: baselinePredictions,
  };
  const comparisons = [];
  for (let index = 0; index < lanes.length; index += 1) {
    comparisons.push(comparePredictions(
      'source-bf16',
      baselinePredictions,
      lanes[index].id,
      lanes[index].predictions.map((row) => row.output)
    ));
    for (let rightIndex = index + 1; rightIndex < lanes.length; rightIndex += 1) {
      comparisons.push(comparePredictions(
        lanes[index].id,
        lanes[index].predictions.map((row) => row.output),
        lanes[rightIndex].id,
        lanes[rightIndex].predictions.map((row) => row.output)
      ));
    }
  }
  const manifestRaw = await readFile(resolve(config.model.modelDir, 'manifest.json'), 'utf8');
  const receipt = {
    schemaVersion: 1,
    artifactKind: 'doppler.translation-quality-benchmark/v1',
    evidenceClass: 'local-constructive',
    suiteId: config.suiteId,
    createdAtUtc: createdAt.toISOString(),
    config: {
      path: config.absoluteConfigPath,
      sha256: config.configHash,
    },
    model: {
      modelId: config.model.modelId,
      modelDir: config.model.modelDir,
      manifestSha256: sha256(manifestRaw),
      sourceCheckpointId: config.model.sourceCheckpointId,
    },
    dataset: {
      path: config.sharedContract.datasetPath,
      sha256: sha256(datasetRaw),
      rows: datasetRows.length,
      pairs: Object.fromEntries(
        [...new Set(datasetRows.map((row) => row.pair))]
          .sort()
          .map((pair) => [pair, datasetRows.filter((row) => row.pair === pair).length])
      ),
    },
    sharedContract: config.sharedContract,
    runtimeProfile: config.runtimeProfile,
    sourceBaseline: {
      id: sourceBaseline.id,
      path: sourceBaseline.path,
      sha256: sourceBaseline.sha256,
      rows: sourceBaseline.predictions.length,
    },
    environment: {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      deviceInfo: lanes[0].deviceInfo,
    },
    metrics,
    comparisons,
    lanes: Object.fromEntries(lanes.map((lane) => [lane.id, {
      label: lane.label,
      runtimeConfig: lane.runtimeConfig,
      executionPlan: lane.executionPlan,
      loadMetrics: lane.loadMetrics,
      warmupOutput: lane.warmupOutput,
      summary: lane.summary,
      dopplerOperatorMetrics: buildDopplerOperatorMetrics(datasetRows, lane.predictions),
      predictionsArtifact: laneArtifacts[lane.id],
    }])),
  };
  const receiptContent = `${JSON.stringify(receipt, null, 2)}\n`;
  const receiptFilename = `${artifactStem}.receipt.json`;
  await writeFile(resolve(config.outputDir, receiptFilename), receiptContent, 'utf8');
  await writeFile(resolve(config.outputDir, 'latest.json'), receiptContent, 'utf8');
  return {
    receipt,
    receiptPath: resolve(config.outputDir, receiptFilename),
  };
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    console.log(usage());
    return;
  }
  if (!args.configPath) {
    throw new Error('--config is required.');
  }
  const config = await loadConfig(args.configPath);
  const datasetRaw = await readFile(config.sharedContract.datasetPath, 'utf8');
  const baselineRaw = await readFile(config.sharedContract.baselinePredictionsPath, 'utf8');
  const datasetRows = validateDatasetRows(
    parseJsonl(datasetRaw),
    config.sharedContract.sampleLimit
  );
  const baselinePredictions = validateBaselinePredictions(
    parseJsonl(baselineRaw).slice(0, datasetRows.length),
    datasetRows
  );
  const lanes = [];
  for (const lane of config.engineLanes) {
    lanes.push(await runLane(config, lane, datasetRows));
  }
  const groups = buildMetricGroups(datasetRows, baselinePredictions, lanes);
  const metrics = await runPythonJson(
    config.metrics.sacrebleuPython,
    SACREBLEU_PROGRAM,
    { groups }
  );
  const output = await writeArtifacts(
    config,
    datasetRaw,
    baselineRaw,
    datasetRows,
    baselinePredictions,
    lanes,
    metrics
  );
  console.log(JSON.stringify({
    ok: true,
    receiptPath: output.receiptPath,
    metrics: output.receipt.metrics,
    comparisons: output.receipt.comparisons,
    laneSummaries: Object.fromEntries(
      Object.entries(output.receipt.lanes).map(([id, lane]) => [id, lane.summary])
    ),
  }, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
});
