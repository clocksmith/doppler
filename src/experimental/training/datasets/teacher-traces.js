import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { parseJsonl } from './jsonl.js';

function isObjectRecord(value) {
  return value && typeof value === 'object' && !Array.isArray(value);
}

function normalizeOptionalString(value) {
  const text = typeof value === 'string' ? value.trim() : '';
  return text || null;
}

function normalizeOptionalStringArray(value, index, label) {
  if (value === undefined || value === null) {
    return null;
  }
  if (!Array.isArray(value)) {
    throw new Error(`teacher trace row ${index + 1} ${label} must be an array of strings.`);
  }
  const normalized = [];
  for (let entryIndex = 0; entryIndex < value.length; entryIndex += 1) {
    const entry = value[entryIndex];
    if (typeof entry !== 'string') {
      throw new Error(`teacher trace row ${index + 1} ${label}[${entryIndex}] must be a string.`);
    }
    const text = entry.trim();
    if (text) normalized.push(text);
  }
  return normalized;
}

function normalizeOptionalObject(value, index, label) {
  if (value === undefined || value === null) {
    return null;
  }
  if (!isObjectRecord(value)) {
    throw new Error(`teacher trace row ${index + 1} ${label} must be an object.`);
  }
  return value;
}

function readStringField(record, names, index, label) {
  for (const name of names) {
    if (Object.prototype.hasOwnProperty.call(record, name)) {
      const value = record[name];
      if (typeof value !== 'string') {
        throw new Error(`teacher trace row ${index + 1} field "${name}" for ${label} must be a string.`);
      }
      const text = value.trim();
      if (!text) {
        throw new Error(`teacher trace row ${index + 1} field "${name}" for ${label} must not be empty.`);
      }
      return { field: name, value };
    }
  }
  return null;
}

function renderMessages(messages, index) {
  if (!Array.isArray(messages) || messages.length === 0) {
    return null;
  }
  const lines = [];
  for (let messageIndex = 0; messageIndex < messages.length; messageIndex += 1) {
    const message = messages[messageIndex];
    if (!isObjectRecord(message)) {
      throw new Error(`teacher trace row ${index + 1} message ${messageIndex + 1} must be an object.`);
    }
    const role = normalizeOptionalString(message.role);
    const content = normalizeOptionalString(message.content);
    if (!role || !content) {
      throw new Error(`teacher trace row ${index + 1} message ${messageIndex + 1} requires role and content.`);
    }
    lines.push(`${role}: ${content}`);
  }
  return lines.join('\n');
}

function resolveTeacherModelId(record, options, index) {
  const teacher = isObjectRecord(record.teacher) ? record.teacher : null;
  const value = normalizeOptionalString(record.teacherModelId)
    || normalizeOptionalString(record.teacher_model_id)
    || normalizeOptionalString(record.teacherModel)
    || normalizeOptionalString(record.teacher_model)
    || normalizeOptionalString(record.model)
    || normalizeOptionalString(teacher?.modelId)
    || normalizeOptionalString(teacher?.model)
    || normalizeOptionalString(options.teacherModelId);
  if (!value) {
    throw new Error(`teacher trace row ${index + 1} requires teacherModelId.`);
  }
  return value;
}

function resolvePrompt(record, index) {
  const direct = readStringField(record, ['prompt', 'source', 'input'], index, 'prompt/source/input');
  if (direct) return direct;
  const rendered = renderMessages(record.messages, index);
  if (rendered) {
    return { field: 'messages', value: rendered };
  }
  throw new Error(`teacher trace row ${index + 1} requires prompt/source/input or messages.`);
}

function resolveCompletion(record, index) {
  const direct = readStringField(record, ['completion', 'target', 'output', 'response'], index, 'completion/target/output');
  if (direct) return direct;
  const assistant = isObjectRecord(record.assistant) ? record.assistant : null;
  const assistantContent = normalizeOptionalString(assistant?.content);
  if (assistantContent) {
    return { field: 'assistant.content', value: assistantContent };
  }
  throw new Error(`teacher trace row ${index + 1} requires completion/target/output/response.`);
}

export function normalizeTeacherTrace(record, index = 0, options = {}) {
  if (!isObjectRecord(record)) {
    throw new Error(`teacher trace row ${index + 1} must be an object.`);
  }
  const prompt = resolvePrompt(record, index);
  const completion = resolveCompletion(record, index);
  const teacherModelId = resolveTeacherModelId(record, options, index);
  const policyId = normalizeOptionalString(record.policyId)
    || normalizeOptionalString(record.policy_id)
    || normalizeOptionalString(record.sourcePolicyId)
    || normalizeOptionalString(record.source_policy_id)
    || normalizeOptionalString(options.policyId)
    || normalizeOptionalString(options.sourcePolicyId);
  const traceId = normalizeOptionalString(record.id)
    || normalizeOptionalString(record.traceId)
    || normalizeOptionalString(record.trace_id)
    || `teacher-trace-${index + 1}`;
  return {
    id: traceId,
    prompt: prompt.value,
    completion: completion.value,
    promptField: prompt.field,
    completionField: completion.field,
    teacherModelId,
    studentBaseModelId: normalizeOptionalString(record.studentBaseModelId)
      || normalizeOptionalString(record.student_base_model_id)
      || normalizeOptionalString(options.studentBaseModelId),
    domain: normalizeOptionalString(record.domain) || normalizeOptionalString(options.domain),
    taskKind: normalizeOptionalString(record.taskKind)
      || normalizeOptionalString(record.task_kind)
      || normalizeOptionalString(options.taskKind),
    policyId,
    sourcePolicyId: policyId,
    gepaCandidateId: normalizeOptionalString(record.gepaCandidateId)
      || normalizeOptionalString(record.gepa_candidate_id)
      || normalizeOptionalString(options.gepaCandidateId),
    sourceFiles: normalizeOptionalStringArray(record.sourceFiles ?? record.source_files ?? options.sourceFiles, index, 'sourceFiles'),
    generationParams: normalizeOptionalObject(record.generationParams ?? record.generation_params ?? options.generationParams, index, 'generationParams'),
    license: normalizeOptionalString(record.license) || normalizeOptionalString(options.license),
    provenance: normalizeOptionalObject(record.provenance ?? options.provenance, index, 'provenance'),
  };
}

export function mapTeacherTraces(records, options = {}) {
  const rows = Array.isArray(records) ? records : [];
  return rows.map((record, index) => normalizeTeacherTrace(record, index, options));
}

export function parseTeacherTraceDataset(text, options = {}) {
  const sourceLabel = String(options.sourceLabel || 'teacher-traces.jsonl');
  const parsed = sourceLabel.endsWith('.json')
    ? JSON.parse(String(text))
    : parseJsonl(String(text));
  if (!Array.isArray(parsed)) {
    throw new Error(`teacher trace dataset "${sourceLabel}" must be a JSON array or JSONL records.`);
  }
  const rows = mapTeacherTraces(parsed, options);
  return {
    sourceLabel,
    rowCount: rows.length,
    rows,
    textPairs: rows.map((row) => ({
      id: row.id,
      prompt: row.prompt,
      completion: row.completion,
      teacherModelId: row.teacherModelId,
      studentBaseModelId: row.studentBaseModelId,
      domain: row.domain,
      taskKind: row.taskKind,
      policyId: row.policyId,
      sourcePolicyId: row.sourcePolicyId,
      gepaCandidateId: row.gepaCandidateId,
      sourceFiles: row.sourceFiles,
      generationParams: row.generationParams,
      license: row.license,
      provenance: row.provenance,
    })),
    lineage: summarizeTeacherTraceLineage(rows),
  };
}

export async function loadTeacherTraceDataset(datasetPath, options = {}) {
  const source = String(datasetPath || '');
  if (!source) {
    throw new Error('loadTeacherTraceDataset requires a dataset path.');
  }
  const absolutePath = resolve(source);
  const text = await (options.readFile ? options.readFile(absolutePath) : readFile(absolutePath, 'utf8'));
  const parsed = parseTeacherTraceDataset(text, {
    ...options,
    sourceLabel: source,
  });
  return {
    absolutePath,
    raw: text,
    ...parsed,
  };
}

export function serializeTeacherTraceTextPairs(rows) {
  if (!Array.isArray(rows)) {
    throw new Error('serializeTeacherTraceTextPairs requires rows.');
  }
  return `${rows.map((row, index) => {
    const normalized = normalizeTeacherTrace(row, index, {
      teacherModelId: row.teacherModelId,
    });
    return JSON.stringify({
      id: normalized.id,
      prompt: normalized.prompt,
      completion: normalized.completion,
      teacherModelId: normalized.teacherModelId,
      studentBaseModelId: normalized.studentBaseModelId,
      domain: normalized.domain,
      taskKind: normalized.taskKind,
      policyId: normalized.policyId,
      sourcePolicyId: normalized.sourcePolicyId,
      gepaCandidateId: normalized.gepaCandidateId,
      sourceFiles: normalized.sourceFiles,
      generationParams: normalized.generationParams,
      license: normalized.license,
      provenance: normalized.provenance,
    });
  }).join('\n')}\n`;
}

export async function writeTeacherTraceTextPairs(inputPath, outputPath, options = {}) {
  const dataset = await loadTeacherTraceDataset(inputPath, options);
  const serialized = serializeTeacherTraceTextPairs(dataset.rows);
  await writeFile(resolve(outputPath), serialized, 'utf8');
  return {
    inputPath: dataset.absolutePath,
    outputPath: resolve(outputPath),
    rowCount: dataset.rowCount,
    lineage: dataset.lineage,
  };
}

export function summarizeTeacherTraceLineage(rows) {
  const teacherModelIds = new Set();
  const studentBaseModelIds = new Set();
  const taskKinds = new Set();
  const sourcePolicyIds = new Set();
  const policyIds = new Set();
  const gepaCandidateIds = new Set();
  for (const row of rows) {
    if (row.teacherModelId) teacherModelIds.add(row.teacherModelId);
    if (row.studentBaseModelId) studentBaseModelIds.add(row.studentBaseModelId);
    if (row.taskKind) taskKinds.add(row.taskKind);
    if (row.policyId) policyIds.add(row.policyId);
    if (row.sourcePolicyId) sourcePolicyIds.add(row.sourcePolicyId);
    if (row.gepaCandidateId) gepaCandidateIds.add(row.gepaCandidateId);
  }
  return {
    teacherModelIds: [...teacherModelIds].sort(),
    studentBaseModelIds: [...studentBaseModelIds].sort(),
    taskKinds: [...taskKinds].sort(),
    policyIds: [...policyIds].sort(),
    sourcePolicyIds: [...sourcePolicyIds].sort(),
    gepaCandidateIds: [...gepaCandidateIds].sort(),
  };
}
