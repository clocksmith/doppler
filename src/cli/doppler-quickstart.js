#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { DOPPLER_VERSION, doppler } from '../index.js';
import { listQuickstartModels, resolveQuickstartModel } from '../client/doppler-registry.js';

const QUICKSTART_CONFIG_PATH = fileURLToPath(new URL('./config/doppler-quickstart.json', import.meta.url));
const JSON_SCHEMA_VERSION = 1;

function usage() {
  return [
    'Usage:',
    '  npx doppler-gpu [prompt] [--model <id>] [--max-tokens <n>] [--temperature <n>] [--json]',
    '  npx doppler-gpu --list-models',
    '  npx doppler-gpu --help',
    '',
    'Examples:',
    '  npx doppler-gpu',
    '  npx doppler-gpu "Summarize WebGPU in one sentence"',
    '  npx doppler-gpu --model gemma3-270m --prompt "Write a haiku about GPUs"',
    '  npx doppler-gpu --list-models',
    '',
    'Notes:',
    '  - First run downloads model files from the quickstart registry.',
    '  - This quickstart bin is separate from the heavier tooling CLI (`doppler`).',
    '  - For benchmark/debug/verify workflows, use `doppler <command> --config ...`.',
  ].join('\n');
}

function asStringOrNull(value) {
  if (value === undefined || value === null) return null;
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  return String(value);
}

function parseNumberFlag(value, label) {
  const normalized = asStringOrNull(value);
  if (normalized === null) return null;
  const parsed = Number(normalized);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be a number.`);
  }
  return parsed;
}

export function parseQuickstartArgs(argv) {
  const flags = {
    model: null,
    prompt: null,
    maxTokens: null,
    temperature: null,
    json: false,
    help: false,
    listModels: false,
  };
  const positional = [];

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--help' || token === '-h') {
      flags.help = true;
      continue;
    }
    if (token === '--json') {
      flags.json = true;
      continue;
    }
    if (token === '--list-models') {
      flags.listModels = true;
      continue;
    }
    if (!token.startsWith('--')) {
      positional.push(token);
      continue;
    }

    const nextValue = argv[index + 1];
    if (nextValue === undefined || nextValue.startsWith('--')) {
      throw new Error(`Missing value for ${token}.`);
    }

    if (token === '--model') {
      flags.model = nextValue;
      index += 1;
      continue;
    }
    if (token === '--prompt') {
      flags.prompt = nextValue;
      index += 1;
      continue;
    }
    if (token === '--max-tokens') {
      flags.maxTokens = nextValue;
      index += 1;
      continue;
    }
    if (token === '--temperature') {
      flags.temperature = nextValue;
      index += 1;
      continue;
    }

    throw new Error(`Unknown flag ${token}.`);
  }

  return {
    ...flags,
    positionalPrompt: positional.length > 0 ? positional.join(' ') : null,
  };
}

export async function readQuickstartConfig() {
  const raw = await fs.readFile(QUICKSTART_CONFIG_PATH, 'utf8');
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('doppler-gpu quickstart config must be a JSON object.');
  }
  if (!parsed.defaults || typeof parsed.defaults !== 'object' || Array.isArray(parsed.defaults)) {
    throw new Error('doppler-gpu quickstart config must define a defaults object.');
  }
  return parsed;
}

function resolvePrompt(parsed, defaults) {
  const positionalPrompt = asStringOrNull(parsed.positionalPrompt);
  if (positionalPrompt !== null) {
    return positionalPrompt;
  }
  const flagPrompt = asStringOrNull(parsed.prompt);
  if (flagPrompt !== null) {
    return flagPrompt;
  }
  return asStringOrNull(defaults.prompt);
}

export async function resolveQuickstartSettings(argv = process.argv.slice(2)) {
  const parsed = parseQuickstartArgs(argv);
  if (parsed.help) {
    return { action: 'help' };
  }
  if (parsed.listModels) {
    return {
      action: 'list-models',
      json: parsed.json === true,
    };
  }

  const config = await readQuickstartConfig();
  const defaults = config.defaults || {};
  const model = asStringOrNull(parsed.model) ?? asStringOrNull(defaults.model);
  const prompt = resolvePrompt(parsed, defaults);
  const maxTokens = parseNumberFlag(parsed.maxTokens, '--max-tokens') ?? parseNumberFlag(defaults.maxTokens, 'quickstart.defaults.maxTokens');
  const temperature = parseNumberFlag(parsed.temperature, '--temperature') ?? parseNumberFlag(defaults.temperature, 'quickstart.defaults.temperature');
  const topK = parseNumberFlag(defaults.topK, 'quickstart.defaults.topK');

  if (model === null) {
    throw new Error('doppler-gpu quickstart requires a default or explicit model id.');
  }
  if (prompt === null) {
    throw new Error('doppler-gpu quickstart requires a default or explicit prompt.');
  }
  if (maxTokens === null) {
    throw new Error('doppler-gpu quickstart requires a default or explicit max token count.');
  }
  if (temperature === null) {
    throw new Error('doppler-gpu quickstart requires a default or explicit temperature.');
  }
  if (topK === null) {
    throw new Error('doppler-gpu quickstart requires a default topK.');
  }

  return {
    action: 'run',
    json: parsed.json === true,
    model,
    prompt,
    maxTokens,
    temperature,
    topK,
  };
}

function formatOneLine(value) {
  return JSON.stringify(String(value ?? '').replace(/\s+/g, ' ').trim());
}

function formatDeviceSummary(deviceInfo) {
  if (!deviceInfo || typeof deviceInfo !== 'object') {
    return 'unknown-device';
  }
  const vendor = asStringOrNull(deviceInfo.vendor) ?? asStringOrNull(deviceInfo.vendorId);
  const architecture = asStringOrNull(deviceInfo.architecture);
  const description = asStringOrNull(deviceInfo.description);
  const parts = [vendor, architecture, description].filter(Boolean);
  return parts.length > 0 ? parts.join(' / ') : 'unknown-device';
}

export async function runQuickstart(settings) {
  const entry = await resolveQuickstartModel(settings.model);
  if (!entry.modes.includes('text')) {
    throw new Error(
      `Quickstart model "${settings.model}" is not text-generative. `
      + 'Use --list-models to see supported text quickstart models.'
    );
  }

  const start = Date.now();
  const model = await doppler.load(settings.model, {
    onProgress: settings.json ? () => {} : undefined,
  });
  try {
    const result = await model.chatText([
      {
        role: 'user',
        content: settings.prompt,
      },
    ], {
      maxTokens: settings.maxTokens,
      temperature: settings.temperature,
      topK: settings.topK,
    });
    return {
      ok: true,
      schemaVersion: JSON_SCHEMA_VERSION,
      version: DOPPLER_VERSION,
      requestedModel: settings.model,
      modelId: model.modelId,
      prompt: settings.prompt,
      content: result.content,
      usage: result.usage,
      elapsedMs: Date.now() - start,
      deviceInfo: model.deviceInfo ?? null,
    };
  } finally {
    await model.unload();
  }
}

async function printModelList(jsonOutput) {
  const models = await listQuickstartModels();
  const textModels = models.filter((entry) => entry.modes.includes('text'));
  if (jsonOutput) {
    console.log(JSON.stringify({ ok: true, schemaVersion: JSON_SCHEMA_VERSION, models: textModels }, null, 2));
    return;
  }
  console.log('Text quickstart models:');
  for (const entry of textModels) {
    const aliasSuffix = entry.aliases.length > 0 ? ` (${entry.aliases.join(', ')})` : '';
    console.log(`- ${entry.modelId}${aliasSuffix}`);
  }
}

function printQuickstartResult(result) {
  console.log(result.content);
  console.error(
    `[doppler-gpu] model=${result.modelId} elapsed=${result.elapsedMs}ms device=${formatDeviceSummary(result.deviceInfo)}`
  );
}

export async function main(argv = process.argv.slice(2)) {
  const settings = await resolveQuickstartSettings(argv);
  if (settings.action === 'help') {
    console.log(usage());
    return;
  }
  if (settings.action === 'list-models') {
    await printModelList(settings.json);
    return;
  }

  const result = await runQuickstart(settings);
  if (settings.json) {
    console.log(JSON.stringify(result, null, 2));
    return;
  }
  console.error(`[doppler-gpu] requestedModel=${settings.model} prompt=${formatOneLine(settings.prompt)}`);
  printQuickstartResult(result);
}

function isMainModule(metaUrl) {
  const entryPath = process.argv[1];
  if (!entryPath) {
    return false;
  }
  return path.resolve(fileURLToPath(metaUrl)) === path.resolve(entryPath);
}

if (isMainModule(import.meta.url)) {
  main().catch((error) => {
    console.error(`[doppler-gpu] ${error?.message || String(error)}`);
    process.exit(1);
  });
}
