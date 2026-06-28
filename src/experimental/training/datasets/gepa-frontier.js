import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

function isObjectRecord(value) {
  return value && typeof value === 'object' && !Array.isArray(value);
}

function normalizeOptionalString(value) {
  const text = typeof value === 'string' ? value.trim() : '';
  return text || null;
}

function normalizeOptionalStringArray(value, label) {
  if (value === undefined || value === null) return null;
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array of strings.`);
  }
  return value.map((entry, index) => {
    if (typeof entry !== 'string') {
      throw new Error(`${label}[${index}] must be a string.`);
    }
    return entry.trim();
  }).filter(Boolean);
}

function normalizeScores(value) {
  if (!isObjectRecord(value)) return {};
  const scores = {};
  for (const [key, raw] of Object.entries(value)) {
    const score = Number(raw);
    if (Number.isFinite(score)) {
      scores[key] = score;
    }
  }
  return scores;
}

function collectCandidateRecords(payload) {
  if (Array.isArray(payload)) return payload;
  if (!isObjectRecord(payload)) {
    throw new Error('GEPA frontier payload must be an array or object.');
  }
  if (Array.isArray(payload.frontier)) return payload.frontier;
  if (Array.isArray(payload.candidates)) return payload.candidates;
  if (Array.isArray(payload.population)) return payload.population;
  if (Array.isArray(payload.prompts)) return payload.prompts;
  throw new Error('GEPA frontier payload requires frontier, candidates, population, or prompts.');
}

function readCandidateContent(record, index) {
  const direct = normalizeOptionalString(record.content)
    || normalizeOptionalString(record.prompt)
    || normalizeOptionalString(record.policy)
    || normalizeOptionalString(record.policyText)
    || normalizeOptionalString(record.systemPrompt);
  if (direct) return direct;
  throw new Error(`GEPA candidate ${index + 1} requires content/prompt/policy text.`);
}

export function normalizeGepaCandidate(record, index = 0, options = {}) {
  const source = isObjectRecord(record?.candidate) ? record.candidate : record;
  if (!isObjectRecord(source)) {
    throw new Error(`GEPA candidate ${index + 1} must be an object.`);
  }
  const id = normalizeOptionalString(source.id)
    || normalizeOptionalString(source.candidateId)
    || `gepa-candidate-${index + 1}`;
  return {
    id,
    content: readCandidateContent(source, index),
    scores: normalizeScores(source.scores ?? record.scores),
    metrics: isObjectRecord(source.metrics ?? record.metrics) ? { ...(source.metrics ?? record.metrics) } : {},
    traces: Array.isArray(source.traces ?? record.traces) ? [...(source.traces ?? record.traces)] : [],
    generation: Number.isFinite(Number(source.generation)) ? Number(source.generation) : null,
    rank: Number.isFinite(Number(source.rank)) ? Number(source.rank) : null,
    dominatedBy: Number.isFinite(Number(source.dominatedBy)) ? Number(source.dominatedBy) : null,
    crowdingDistance: Number.isFinite(Number(source.crowdingDistance)) ? Number(source.crowdingDistance) : null,
    targetType: normalizeOptionalString(source.targetType) || normalizeOptionalString(options.targetType),
    sourcePolicyId: normalizeOptionalString(source.sourcePolicyId)
      || normalizeOptionalString(record.sourcePolicyId)
      || normalizeOptionalString(options.sourcePolicyId)
      || `gepa:${id}`,
  };
}

export function parseGepaFrontier(text, options = {}) {
  const payload = JSON.parse(String(text));
  const candidates = collectCandidateRecords(payload).map((record, index) => (
    normalizeGepaCandidate(record, index, options)
  ));
  return {
    candidates,
    lineage: summarizeGepaFrontier(candidates),
  };
}

export async function loadGepaFrontier(frontierPath, options = {}) {
  const absolutePath = resolve(String(frontierPath || ''));
  if (!frontierPath) {
    throw new Error('loadGepaFrontier requires a frontier path.');
  }
  const raw = await (options.readFile ? options.readFile(absolutePath) : readFile(absolutePath, 'utf8'));
  const parsed = parseGepaFrontier(raw, options);
  return {
    absolutePath,
    raw,
    ...parsed,
  };
}

function normalizeTraceInput(trace, traceIndex, candidateId) {
  const input = normalizeOptionalString(trace.input)
    || normalizeOptionalString(trace.prompt)
    || normalizeOptionalString(trace.task?.input);
  if (!input) {
    throw new Error(`GEPA candidate ${candidateId} trace ${traceIndex + 1} requires input.`);
  }
  return input;
}

function normalizeTraceOutput(trace, traceIndex, candidateId) {
  const output = normalizeOptionalString(trace.actualOutput)
    || normalizeOptionalString(trace.output)
    || normalizeOptionalString(trace.completion)
    || normalizeOptionalString(trace.response);
  if (!output) {
    throw new Error(`GEPA candidate ${candidateId} trace ${traceIndex + 1} requires output/actualOutput.`);
  }
  return output;
}

export function buildTeacherTracesFromGepaFrontier(candidates, options = {}) {
  const teacherModelId = normalizeOptionalString(options.teacherModelId);
  if (!teacherModelId) {
    throw new Error('buildTeacherTracesFromGepaFrontier requires teacherModelId.');
  }
  const includeFailures = options.includeFailures === true;
  const rows = [];
  for (const candidate of candidates) {
    const traces = Array.isArray(candidate.traces) ? candidate.traces : [];
    for (let traceIndex = 0; traceIndex < traces.length; traceIndex += 1) {
      const trace = traces[traceIndex];
      if (!isObjectRecord(trace)) {
        throw new Error(`GEPA candidate ${candidate.id} trace ${traceIndex + 1} must be an object.`);
      }
      if (trace.success === false && !includeFailures) {
        continue;
      }
      const taskInput = normalizeTraceInput(trace, traceIndex, candidate.id);
      const completion = normalizeTraceOutput(trace, traceIndex, candidate.id);
      rows.push({
        schemaVersion: 1,
        artifactType: 'teacher_trace',
        traceFormat: 'doppler_teacher_trace_v1',
        id: `${candidate.id}-trace-${traceIndex + 1}`,
        teacherModel: teacherModelId,
        teacherModelId,
        studentBaseModelId: normalizeOptionalString(options.studentBaseModelId),
        domain: normalizeOptionalString(options.domain),
        taskKind: normalizeOptionalString(options.taskKind) || normalizeOptionalString(trace.taskKind),
        policyId: candidate.sourcePolicyId,
        sourcePolicyId: candidate.sourcePolicyId,
        gepaCandidateId: candidate.id,
        sourceFiles: normalizeOptionalStringArray(trace.sourceFiles ?? trace.source_files ?? options.sourceFiles, 'GEPA trace sourceFiles'),
        generationParams: isObjectRecord(trace.generationParams ?? trace.generation_params)
          ? { ...(trace.generationParams ?? trace.generation_params) }
          : null,
        license: normalizeOptionalString(trace.license) || normalizeOptionalString(options.license),
        prompt: `${candidate.content}\n\nTask:\n${taskInput}`,
        completion,
        provenance: {
          provider: 'gepa-frontier',
          candidateId: candidate.id,
          traceIndex,
        },
        scores: candidate.scores,
        metrics: candidate.metrics,
      });
    }
  }
  return rows;
}

export function serializeGepaTeacherTraces(rows) {
  if (!Array.isArray(rows)) {
    throw new Error('serializeGepaTeacherTraces requires rows.');
  }
  return `${rows.map((row) => JSON.stringify(row)).join('\n')}\n`;
}

export async function writeGepaTeacherTraces(frontierPath, outputPath, options = {}) {
  const frontier = await loadGepaFrontier(frontierPath, options);
  const rows = buildTeacherTracesFromGepaFrontier(frontier.candidates, options);
  await writeFile(resolve(outputPath), serializeGepaTeacherTraces(rows), 'utf8');
  return {
    inputPath: frontier.absolutePath,
    outputPath: resolve(outputPath),
    candidateCount: frontier.candidates.length,
    rowCount: rows.length,
    lineage: frontier.lineage,
  };
}

export function summarizeGepaFrontier(candidates) {
  const candidateIds = new Set();
  const sourcePolicyIds = new Set();
  const objectiveNames = new Set();
  for (const candidate of candidates) {
    candidateIds.add(candidate.id);
    if (candidate.sourcePolicyId) sourcePolicyIds.add(candidate.sourcePolicyId);
    for (const objective of Object.keys(candidate.scores || {})) {
      objectiveNames.add(objective);
    }
  }
  return {
    candidateIds: [...candidateIds].sort(),
    sourcePolicyIds: [...sourcePolicyIds].sort(),
    objectiveNames: [...objectiveNames].sort(),
  };
}
