#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

import { parseJsonl } from '../src/experimental/training/datasets/jsonl.js';

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) {
      throw new Error(`Unexpected positional argument "${token}".`);
    }
    const key = token.slice(2);
    const next = argv[index + 1];
    if (next === undefined || next.startsWith('--')) {
      args[key] = true;
    } else {
      args[key] = next;
      index += 1;
    }
  }
  return args;
}

function requireString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${label} is required.`);
  }
  return value.trim();
}

function optionalString(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function parseStringList(value, label) {
  if (value === undefined || value === null || value === '') {
    return [];
  }
  if (Array.isArray(value)) {
    return value.map((entry, index) => {
      if (typeof entry !== 'string' || !entry.trim()) {
        throw new Error(`${label}[${index}] must be a non-empty string.`);
      }
      return entry.trim();
    });
  }
  if (typeof value !== 'string') {
    throw new Error(`${label} must be a comma-separated string or an array of strings.`);
  }
  return String(value)
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function normalizeRow(record, index) {
  if (!record || typeof record !== 'object' || Array.isArray(record)) {
    throw new Error(`Teacher seed row ${index + 1} must be an object.`);
  }
  const prompt = record.prompt ?? record.source ?? record.input;
  if (typeof prompt !== 'string' || !prompt.trim()) {
    throw new Error(`Teacher seed row ${index + 1} requires prompt/source/input.`);
  }
  return {
    id: String(record.id || record.rowId || `trace-${index + 1}`),
    prompt: prompt.trim(),
    taskKind: optionalString(record.taskKind),
    domain: optionalString(record.domain),
    policyId: optionalString(record.policyId) || optionalString(record.sourcePolicyId),
    sourceFiles: parseStringList(record.sourceFiles ?? record.source_files, `Teacher seed row ${index + 1} sourceFiles`),
    license: optionalString(record.license),
    provenance: record.provenance && typeof record.provenance === 'object' && !Array.isArray(record.provenance)
      ? record.provenance
      : null,
  };
}

async function loadSeedRows(inputPath, limit = null) {
  const absolutePath = resolve(String(inputPath));
  const raw = await readFile(absolutePath, 'utf8');
  const parsed = absolutePath.endsWith('.json')
    ? JSON.parse(raw)
    : parseJsonl(raw);
  if (!Array.isArray(parsed)) {
    throw new Error(`Teacher seed input "${absolutePath}" must be a JSON array or JSONL rows.`);
  }
  const rows = parsed.map((row, index) => normalizeRow(row, index));
  return Number.isInteger(limit) && limit > 0 ? rows.slice(0, limit) : rows;
}

function buildChatBody(model, row, options) {
  const systemPrompt = typeof options.systemPrompt === 'string' && options.systemPrompt.trim()
    ? options.systemPrompt.trim()
    : 'You are generating concise, correct Doppler/Reploid code-agent training traces.';
  return {
    model,
    temperature: options.temperature,
    max_tokens: options.maxTokens,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: row.prompt },
    ],
  };
}

async function callTeacher(row, options) {
  const body = buildChatBody(options.model, row, options);
  const response = await fetch(`${options.baseUrl.replace(/\/$/, '')}/chat/completions`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${options.apiKey}`,
    },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`Teacher request failed for row "${row.id}" with HTTP ${response.status}: ${text.slice(0, 500)}`);
  }
  const payload = JSON.parse(text);
  const completion = payload.choices?.[0]?.message?.content;
  if (typeof completion !== 'string' || !completion.trim()) {
    throw new Error(`Teacher response for row "${row.id}" did not include choices[0].message.content.`);
  }
  return completion.trim();
}

function serializeTrace(row, completion, options) {
  const policyId = row.policyId || options.policyId;
  const sourceFiles = row.sourceFiles.length > 0 ? row.sourceFiles : options.sourceFiles;
  return JSON.stringify({
    schemaVersion: 1,
    artifactType: 'teacher_trace',
    traceFormat: 'doppler_teacher_trace_v1',
    id: row.id,
    teacherModel: options.model,
    teacherModelId: options.model,
    studentBaseModelId: options.studentBaseModelId,
    domain: row.domain || options.domain,
    taskKind: row.taskKind || options.taskKind,
    policyId,
    sourcePolicyId: policyId,
    sourceFiles,
    generationParams: {
      temperature: options.temperature,
      maxTokens: options.maxTokens,
      systemPrompt: options.systemPrompt,
      endpoint: 'chat/completions',
    },
    license: row.license || options.license,
    prompt: row.prompt,
    completion,
    provenance: {
      ...(row.provenance || {}),
      provider: options.provider,
      baseUrl: options.baseUrl,
      inputPath: options.inputPath,
      seedId: row.id,
      sourceFiles,
      generatedAt: new Date().toISOString(),
    },
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = requireString(args.input, '--input');
  const outputPath = resolve(requireString(args.out, '--out'));
  const model = requireString(args.model, '--model');
  const baseUrl = requireString(args['base-url'], '--base-url');
  const apiKeyEnv = requireString(args['api-key-env'], '--api-key-env');
  const apiKey = process.env[apiKeyEnv];
  if (typeof apiKey !== 'string' || !apiKey.trim()) {
    throw new Error(`Environment variable ${apiKeyEnv} is required for teacher trace generation.`);
  }
  const options = {
    model,
    baseUrl,
    apiKey: apiKey.trim(),
    provider: String(args.provider || 'openai-compatible').trim(),
    studentBaseModelId: String(args['student-base-model-id'] || '').trim() || null,
    domain: String(args.domain || '').trim() || null,
    taskKind: String(args['task-kind'] || '').trim() || null,
    policyId: String(args['policy-id'] || args['source-policy-id'] || '').trim() || null,
    sourceFiles: parseStringList(args['source-files'], '--source-files'),
    license: String(args.license || '').trim() || null,
    inputPath: resolve(inputPath),
    systemPrompt: typeof args['system-prompt'] === 'string' ? args['system-prompt'] : null,
    temperature: Number.isFinite(Number(args.temperature)) ? Number(args.temperature) : 0.2,
    maxTokens: Number.isInteger(Number(args['max-tokens'])) ? Number(args['max-tokens']) : 512,
  };
  const limit = Number.isInteger(Number(args.limit)) ? Number(args.limit) : null;
  const rows = await loadSeedRows(inputPath, limit);
  const traces = [];
  for (const row of rows) {
    const completion = await callTeacher(row, options);
    traces.push(serializeTrace(row, completion, options));
  }
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${traces.join('\n')}\n`, 'utf8');
  console.log(JSON.stringify({
    ok: true,
    inputPath: resolve(inputPath),
    outputPath,
    rowCount: traces.length,
    teacherModelId: model,
    studentBaseModelId: options.studentBaseModelId,
  }, null, 2));
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
