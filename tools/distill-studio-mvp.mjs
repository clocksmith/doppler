#!/usr/bin/env node

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

function parseArgs(argv) {
  const parsed = {
    mode: null,
    teacher: null,
    student: null,
    holdout: null,
    out: 'reports/distill-studio/mvp-output.json',
  };
  const rest = argv.slice(2);
  if (rest.length === 0) {
    throw new Error('Usage: node tools/distill-studio-mvp.mjs <replay-teacher|branch-compare|mini-eval> [flags]');
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
    return { absolute, value: JSON.parse(raw) };
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
  return fromResults;
}

function buildReplay(teacherReport) {
  const metrics = resolveTrainingMetrics(teacherReport);
  return {
    schemaVersion: 1,
    mode: 'replay-teacher',
    generatedAt: new Date().toISOString(),
    timeline: metrics.map((entry) => ({
      step: entry.step ?? null,
      objective: entry.objective ?? null,
      total_loss: entry.total_loss ?? null,
      step_time_ms: entry.step_time_ms ?? null,
      telemetry_alerts: entry.telemetry_alerts ?? [],
    })),
  };
}

function buildBranchCompare(teacherReport, studentReport) {
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
    teacherSteps: teacher.length,
    studentSteps: student.length,
    comparedSteps: deltas.length,
    avgLossDelta: avgDelta,
    deltas,
  };
}

function buildMiniEval(teacherReport, studentReport, holdoutRows) {
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
    holdoutSize: holdoutRows.length,
    teacherAvgLoss: avg(teacherLoss),
    studentAvgLoss: avg(studentLoss),
    pulsePass: Number.isFinite(avg(studentLoss))
      ? avg(studentLoss) <= (Number.isFinite(avg(teacherLoss)) ? avg(teacherLoss) * 1.25 : avg(studentLoss))
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
  const teacher = (await readJson(args.teacher, '--teacher')).value;
  const student = args.student ? (await readJson(args.student, '--student')).value : null;
  const holdout = await readHoldout(args.holdout);

  let output;
  if (args.mode === 'replay-teacher') {
    output = buildReplay(teacher);
  } else if (args.mode === 'branch-compare') {
    if (!student) {
      throw new Error('--student is required for branch-compare.');
    }
    output = buildBranchCompare(teacher, student);
  } else if (args.mode === 'mini-eval') {
    if (!student) {
      throw new Error('--student is required for mini-eval.');
    }
    output = buildMiniEval(teacher, student, holdout);
  } else {
    throw new Error(`Unknown mode: ${args.mode}`);
  }

  const outPath = resolve(args.out);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
  console.log(JSON.stringify({ ok: true, outPath }, null, 2));
}

await main();
