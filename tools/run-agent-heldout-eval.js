#!/usr/bin/env node

import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { tmpdir } from 'node:os';

import { parseJsonl } from '../src/experimental/training/datasets/jsonl.js';
import { loadEvalDataset } from '../src/experimental/training/operator-eval.js';
import { evaluateAgentHeldoutRows } from '../src/experimental/training/operator-agent-eval.js';
import { loadTrainingWorkloadPack } from '../src/experimental/training/workloads.js';
import { sha256Hex } from '../src/utils/sha256.js';

function parseArgs(argv) {
  const args = {
    workload: null,
    evalDatasetId: null,
    dataset: null,
    candidates: null,
    policy: null,
    out: null,
    patchRoot: null,
    checkpointId: null,
    checkpointStep: null,
    stage: null,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const token = String(argv[index] || '');
    const readValue = () => {
      const value = argv[index + 1];
      if (value === undefined || String(value).startsWith('--')) {
        throw new Error(`${token} requires a value.`);
      }
      index += 1;
      return String(value);
    };
    if (token === '--workload') args.workload = readValue();
    else if (token === '--eval-dataset-id') args.evalDatasetId = readValue();
    else if (token === '--dataset') args.dataset = readValue();
    else if (token === '--candidates') args.candidates = readValue();
    else if (token === '--policy') args.policy = readValue();
    else if (token === '--out') args.out = readValue();
    else if (token === '--patch-root') args.patchRoot = readValue();
    else if (token === '--checkpoint-id') args.checkpointId = readValue();
    else if (token === '--checkpoint-step') args.checkpointStep = Number(readValue());
    else if (token === '--stage') args.stage = readValue();
    else throw new Error(`Unknown argument: ${token}`);
  }
  if (!args.candidates) throw new Error('--candidates is required.');
  if (!args.out) throw new Error('--out is required.');
  if (!args.workload && !args.dataset) {
    throw new Error('--workload or --dataset is required.');
  }
  return args;
}

function parseRows(text, sourceLabel) {
  const rows = sourceLabel.endsWith('.json')
    ? JSON.parse(text)
    : parseJsonl(text);
  if (!Array.isArray(rows)) {
    throw new Error(`${sourceLabel} must be a JSON array or JSONL records.`);
  }
  return rows;
}

async function readRows(pathValue) {
  const absolutePath = resolve(String(pathValue));
  const raw = await readFile(absolutePath, 'utf8');
  return {
    absolutePath,
    raw,
    rows: parseRows(raw, absolutePath),
  };
}

async function readPolicy(policyInput) {
  if (!policyInput) return null;
  const text = String(policyInput).trim();
  if (text.startsWith('{')) {
    return JSON.parse(text);
  }
  const raw = await readFile(resolve(text), 'utf8');
  return JSON.parse(raw);
}

function resolveEvalDataset(workload, evalDatasetId) {
  const evalDatasets = Array.isArray(workload?.evalDatasets) ? workload.evalDatasets : [];
  if (evalDatasetId) {
    const evalDataset = evalDatasets.find((entry) => entry.id === evalDatasetId);
    if (!evalDataset) {
      throw new Error(`Workload does not declare eval dataset "${evalDatasetId}".`);
    }
    return evalDataset;
  }
  const evalDataset = evalDatasets.find((entry) => entry.agentEval);
  if (!evalDataset) {
    throw new Error('Workload does not declare an eval dataset with agentEval.');
  }
  return evalDataset;
}

function resolveRowId(row, index) {
  const id = typeof row?.id === 'string' && row.id.trim()
    ? row.id.trim()
    : (typeof row?.rowId === 'string' && row.rowId.trim() ? row.rowId.trim() : null);
  return id || `row-${index + 1}`;
}

function readCompletion(row) {
  for (const key of ['completion', 'output', 'response', 'candidate']) {
    if (typeof row?.[key] === 'string') {
      return row[key];
    }
  }
  if (row?.assistant && typeof row.assistant.content === 'string') {
    return row.assistant.content;
  }
  return '';
}

function buildCandidateMap(candidateRows) {
  const byId = new Map();
  for (let index = 0; index < candidateRows.length; index += 1) {
    const row = candidateRows[index];
    byId.set(resolveRowId(row, index), row);
  }
  return byId;
}

function extractFencedDiff(text) {
  const match = String(text || '').match(/```(?:diff|patch)?\s*\n([\s\S]*?)```/i);
  return match ? match[1].trim() : null;
}

function extractUnifiedDiff(text) {
  const fenced = extractFencedDiff(text);
  if (fenced) return fenced;
  const lines = String(text || '').split(/\r?\n/);
  const start = lines.findIndex((line) => line.startsWith('diff --git ') || line.startsWith('--- '));
  return start >= 0 ? lines.slice(start).join('\n').trim() : null;
}

function rowRequiresPatch(row, policy) {
  return policy?.requirePatchApplies === true || row?.agentEval?.requirePatchApplies === true;
}

async function buildPatchStatuses(datasetRows, candidateRows, policy, patchRoot) {
  const requiresPatch = datasetRows.some((row) => rowRequiresPatch(row, policy));
  if (!requiresPatch) return {};
  if (!patchRoot) {
    throw new Error('--patch-root is required when agentEval requires patch application evidence.');
  }
  const candidateMap = buildCandidateMap(candidateRows);
  const tempRoot = await mkdtemp(join(tmpdir(), 'doppler-agent-heldout-eval-'));
  const statuses = {};
  try {
    for (let index = 0; index < datasetRows.length; index += 1) {
      const row = datasetRows[index];
      if (!rowRequiresPatch(row, policy)) continue;
      const rowId = resolveRowId(row, index);
      const candidate = candidateMap.get(rowId) || null;
      const diff = extractUnifiedDiff(readCompletion(candidate));
      if (!diff) {
        statuses[rowId] = {
          applies: false,
          error: 'Candidate did not include a unified diff.',
        };
        continue;
      }
      const patchPath = join(tempRoot, `${rowId.replace(/[^A-Za-z0-9_-]/g, '_')}.patch`);
      await writeFile(patchPath, `${diff}\n`, 'utf8');
      const result = spawnSync('git', ['-C', resolve(patchRoot), 'apply', '--check', patchPath], {
        encoding: 'utf8',
      });
      statuses[rowId] = {
        applies: result.status === 0,
        patchPath,
        stdout: result.stdout || null,
        stderr: result.stderr || null,
        error: result.status === 0 ? null : (result.stderr || result.stdout || `git apply exited ${result.status}`),
      };
    }
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
  return statuses;
}

function buildReportPayload(context) {
  const agentEval = context.agentEval;
  return {
    artifactType: 'training_eval_report',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    workloadId: context.workload?.id || null,
    workloadPath: context.loadedWorkload?.absolutePath || null,
    workloadSha256: context.loadedWorkload?.workloadSha256 || null,
    configHash: context.workload?.configHash || null,
    datasetPath: context.datasetPath,
    datasetHash: sha256Hex(context.datasetRaw),
    baseModelId: context.workload?.baseModelId || null,
    baseModelRef: context.workload?.baseModelId || null,
    stage: context.stage || 'agent_heldout_eval',
    checkpointStep: context.checkpointStep ?? null,
    checkpointId: context.checkpointId || null,
    evalDatasetId: context.evalDatasetId,
    metrics: {
      agent_heldout_gate: {
        score: agentEval.passRate,
        samples: agentEval.totalRows,
        passed: agentEval.passed,
      },
    },
    primaryMetric: 'agent_heldout_pass_rate',
    primaryScore: agentEval.passRate,
    loss: null,
    baseline: null,
    qualityClaim: {
      metric: 'agent_heldout_pass_rate',
      selectionGoal: 'max',
      adapterScore: agentEval.passRate,
      baselineScore: null,
      delta: null,
      absoluteImprovement: null,
      relativeImprovement: null,
      minAbsoluteImprovement: null,
      minRelativeImprovement: null,
      improved: agentEval.passed,
      requireImprovement: false,
    },
    agentEval,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const loadedWorkload = args.workload ? await loadTrainingWorkloadPack(args.workload) : null;
  const workload = loadedWorkload?.workload || null;
  const evalDataset = workload ? resolveEvalDataset(workload, args.evalDatasetId) : null;
  const policy = await readPolicy(args.policy) || evalDataset?.agentEval || null;
  if (!policy) {
    throw new Error('agent held-out eval requires --policy or a workload evalDataset.agentEval block.');
  }
  const datasetPath = args.dataset || evalDataset?.datasetPath;
  if (!datasetPath) {
    throw new Error('agent held-out eval requires --dataset or a workload eval dataset path.');
  }
  const dataset = await loadEvalDataset(datasetPath);
  const candidates = await readRows(args.candidates);
  const patchStatuses = await buildPatchStatuses(dataset.rows, candidates.rows, policy, args.patchRoot);
  const agentEval = evaluateAgentHeldoutRows(dataset.rows, candidates.rows, {
    policy,
    patchStatuses,
  });
  const payload = buildReportPayload({
    loadedWorkload,
    workload,
    datasetPath,
    datasetRaw: dataset.raw,
    evalDatasetId: args.evalDatasetId || evalDataset?.id || 'agent-heldout-eval',
    stage: args.stage,
    checkpointId: args.checkpointId,
    checkpointStep: Number.isFinite(args.checkpointStep) ? args.checkpointStep : null,
    agentEval,
  });
  const outPath = resolve(args.out);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  console.log(JSON.stringify({
    ok: agentEval.passed,
    reportPath: outPath,
    evalDatasetId: payload.evalDatasetId,
    passRate: agentEval.passRate,
    passedRows: agentEval.passedRows,
    totalRows: agentEval.totalRows,
  }, null, 2));
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
