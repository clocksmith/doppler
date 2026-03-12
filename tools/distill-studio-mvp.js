#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const STUDENT_LOSS_TOLERANCE = 1.25;

function parseArgs(argv) {
  const parsed = {
    mode: null,
    teacher: null,
    student: null,
    holdout: null,
    workloadPack: null,
    out: 'reports/distill-studio/mvp-output.json',
  };
  const rest = argv.slice(2);
  if (rest.length === 0) {
    throw new Error('Usage: node tools/distill-studio-mvp.js <replay-teacher|branch-compare|mini-eval> [--teacher <path>] [--student <path>] [--holdout <path>] [--workload-pack <path>] [--out <path>]');
  }
  parsed.mode = rest[0];
  for (let i = 1; i < rest.length; i += 1) {
    const arg = rest[i];
    if (arg === '--teacher') {
      parsed.teacher = rest[i + 1] || null;
      i += 1;
      continue;
    }
    if (arg === '--student') {
      parsed.student = rest[i + 1] || null;
      i += 1;
      continue;
    }
    if (arg === '--holdout') {
      parsed.holdout = rest[i + 1] || null;
      i += 1;
      continue;
    }
    if (arg === '--workload-pack') {
      parsed.workloadPack = rest[i + 1] || null;
      i += 1;
      continue;
    }
    if (arg === '--out') {
      parsed.out = rest[i + 1] || parsed.out;
      i += 1;
      continue;
    }
    throw new Error(`Unknown flag: ${arg}`);
  }
  return parsed;
}

async function readJson(pathValue, label) {
  const absolute = resolve(String(pathValue));
  const raw = await readFile(absolute, 'utf8');
  try {
    return { absolute, raw, value: JSON.parse(raw) };
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
}

function resolveTrainingMetrics(report) {
  const direct = Array.isArray(report?.metrics?.trainingMetricsReport)
    ? report.metrics.trainingMetricsReport
    : null;
  if (direct) return direct;
  const fromResults = Array.isArray(report?.results)
    ? report.results.flatMap((entry) => (Array.isArray(entry?.metrics?.trainingMetricsReport)
      ? entry.metrics.trainingMetricsReport
      : []))
    : [];
  if (fromResults.length > 0) {
    console.warn('[distill-studio] resolveTrainingMetrics: using fallback results[] path; report.metrics.trainingMetricsReport was absent.');
  }
  return fromResults;
}

function computeReportId(reportObj, rawJson) {
  const explicit = String(reportObj?.reportId || reportObj?.id || reportObj?.report?.id || '').trim();
  if (explicit) return explicit;
  return createHash('sha256').update(rawJson).digest('hex');
}

function resolveWorkloadPackTraceability(workloadPack) {
  if (!workloadPack) return null;
  const id = String(workloadPack.value?.id || '').trim();
  if (!id) {
    throw new Error('--workload-pack must include a non-empty "id" field.');
  }
  return {
    id,
    path: workloadPack.absolute,
    sha256: createHash('sha256').update(workloadPack.raw).digest('hex'),
  };
}

function buildTraceability({ teacher, student, workloadPack }) {
  const traceability = {
    teacherReportId: computeReportId(teacher.value, teacher.raw),
    studentReportId: student ? computeReportId(student.value, student.raw) : null,
    workloadPack: resolveWorkloadPackTraceability(workloadPack),
  };
  return traceability;
}

function buildReplay(teacherReport, traceability) {
  const metrics = resolveTrainingMetrics(teacherReport);
  return {
    schemaVersion: 1,
    mode: 'replay-teacher',
    generatedAt: new Date().toISOString(),
    traceability,
    timeline: metrics.map((entry) => ({
      step: entry.step ?? null,
      objective: entry.objective ?? null,
      total_loss: entry.total_loss ?? null,
      step_time_ms: entry.step_time_ms ?? null,
      telemetry_alerts: entry.telemetry_alerts ?? [],
    })),
  };
}

function buildBranchCompare(teacherReport, studentReport, traceability) {
  const teacher = resolveTrainingMetrics(teacherReport);
  const student = resolveTrainingMetrics(studentReport);
  const count = Math.min(teacher.length, student.length);
  const deltas = [];
  for (let i = 0; i < count; i += 1) {
    const t = Number(teacher[i]?.total_loss);
    const s = Number(student[i]?.total_loss);
    if (!Number.isFinite(t) || !Number.isFinite(s)) continue;
    deltas.push({
      step: i + 1,
      teacherLoss: t,
      studentLoss: s,
      delta: s - t,
    });
  }
  const avgDelta = deltas.length
    ? deltas.reduce((acc, row) => acc + row.delta, 0) / deltas.length
    : null;
  return {
    schemaVersion: 1,
    mode: 'branch-compare',
    generatedAt: new Date().toISOString(),
    traceability,
    teacherSteps: teacher.length,
    studentSteps: student.length,
    comparedSteps: deltas.length,
    avgLossDelta: avgDelta,
    deltas,
  };
}

function buildMiniEval(teacherReport, studentReport, holdoutRows, traceability) {
  const teacher = resolveTrainingMetrics(teacherReport);
  const student = resolveTrainingMetrics(studentReport);
  const pulseSize = Math.max(1, Math.floor(holdoutRows.length || 1));
  const teacherLoss = teacher.slice(-pulseSize).map((entry) => Number(entry?.total_loss)).filter(Number.isFinite);
  const studentLoss = student.slice(-pulseSize).map((entry) => Number(entry?.total_loss)).filter(Number.isFinite);
  const avg = (rows) => (rows.length ? rows.reduce((acc, value) => acc + value, 0) / rows.length : null);
  return {
    schemaVersion: 1,
    mode: 'mini-eval',
    generatedAt: new Date().toISOString(),
    traceability,
    holdoutSize: holdoutRows.length,
    teacherAvgLoss: avg(teacherLoss),
    studentAvgLoss: avg(studentLoss),
    pulsePass: Number.isFinite(avg(studentLoss))
      ? avg(studentLoss) <= (Number.isFinite(avg(teacherLoss)) ? avg(teacherLoss) * STUDENT_LOSS_TOLERANCE : avg(studentLoss))
      : false,
  };
}

async function readHoldout(pathValue) {
  if (!pathValue) return [];
  const { value } = await readJson(pathValue, '--holdout');
  if (!Array.isArray(value)) {
    throw new Error('--holdout must be a JSON array.');
  }
  return value;
}

async function main() {
  const args = parseArgs(process.argv);
  if (!args.teacher) {
    throw new Error('--teacher is required.');
  }
  const teacher = await readJson(args.teacher, '--teacher');
  const student = args.student ? await readJson(args.student, '--student') : null;
  const workloadPack = args.workloadPack ? await readJson(args.workloadPack, '--workload-pack') : null;
  const holdout = await readHoldout(args.holdout);
  const traceability = buildTraceability({ teacher, student, workloadPack });

  let output;
  if (args.mode === 'replay-teacher') {
    output = buildReplay(teacher.value, traceability);
  } else if (args.mode === 'branch-compare') {
    if (!student) {
      throw new Error('--student is required for branch-compare.');
    }
    output = buildBranchCompare(teacher.value, student.value, traceability);
  } else if (args.mode === 'mini-eval') {
    if (!student) {
      throw new Error('--student is required for mini-eval.');
    }
    output = buildMiniEval(teacher.value, student.value, holdout, traceability);
  } else {
    throw new Error(`Unknown mode: ${args.mode}`);
  }

  const outPath = resolve(args.out);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
  console.log(JSON.stringify({ ok: true, outPath }, null, 2));
}

await main();
