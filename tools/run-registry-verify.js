#!/usr/bin/env node

import { spawn } from 'node:child_process';

const DEFAULT_REGISTRY_URL = process.env.DOPPLER_HF_REGISTRY_URL
  || 'https://huggingface.co/Clocksmith/rdrr/resolve/main/registry/catalog.json';

function usage() {
  console.error(
    'Usage: node tools/run-registry-verify.js <model-alias-or-id> [--registry-url <url>] [--surface <auto|node|browser>]'
  );
}

function normalizeToken(value) {
  return typeof value === 'string' ? value.trim().toLowerCase() : '';
}

function normalizePath(pathValue) {
  if (typeof pathValue !== 'string') return '';
  return pathValue.trim().replace(/^\/+/, '');
}

function resolveModelUrl(entry) {
  const hf = entry?.hf;
  if (hf && typeof hf === 'object') {
    const repoId = typeof hf.repoId === 'string' ? hf.repoId.trim() : '';
    const revision = typeof hf.revision === 'string' ? hf.revision.trim() : '';
    const repoPath = normalizePath(hf.path);
    if (repoId && revision && repoPath) {
      return `https://huggingface.co/${repoId}/resolve/${encodeURIComponent(revision)}/${repoPath}`;
    }
  }

  const baseUrl = typeof entry?.baseUrl === 'string' ? entry.baseUrl.trim() : '';
  if (/^https?:\/\//.test(baseUrl)) {
    return baseUrl.replace(/\/+$/, '');
  }
  throw new Error(`Registry entry "${entry?.modelId || 'unknown'}" does not expose a remote HF URL.`);
}

function findRegistryEntry(registry, token) {
  const models = Array.isArray(registry?.models) ? registry.models : [];
  const needle = normalizeToken(token);
  if (!needle) return null;
  for (const model of models) {
    if (normalizeToken(model?.modelId) === needle) {
      return model;
    }
    const aliases = Array.isArray(model?.aliases) ? model.aliases : [];
    for (const alias of aliases) {
      if (normalizeToken(alias) === needle) {
        return model;
      }
    }
  }
  return null;
}

function parseArgs(argv) {
  const out = {
    model: '',
    registryUrl: DEFAULT_REGISTRY_URL,
    surface: 'auto',
  };

  const positional = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--registry-url') {
      out.registryUrl = argv[i + 1] ? String(argv[i + 1]).trim() : '';
      i += 1;
      continue;
    }
    if (arg === '--surface') {
      out.surface = argv[i + 1] ? String(argv[i + 1]).trim() : '';
      i += 1;
      continue;
    }
    positional.push(arg);
  }

  out.model = positional[0] ? String(positional[0]).trim() : '';
  return out;
}

async function loadRegistry(registryUrl) {
  if (!registryUrl) {
    throw new Error('Registry URL is required.');
  }
  const response = await fetch(registryUrl);
  if (!response.ok) {
    throw new Error(`Failed to load registry ${registryUrl}: HTTP ${response.status}.`);
  }
  const payload = await response.json();
  if (!payload || typeof payload !== 'object' || !Array.isArray(payload.models)) {
    throw new Error(`Registry payload is invalid at ${registryUrl}.`);
  }
  return payload;
}

function runVerify(config, surface) {
  const args = [
    'tools/doppler-cli.js',
    'verify',
    '--config',
    JSON.stringify(config),
    '--json',
  ];
  if (surface) {
    args.push('--surface', surface);
  }

  const child = spawn('node', args, { stdio: 'inherit', env: process.env });
  child.on('exit', (code) => {
    process.exit(code ?? 1);
  });
  child.on('error', (error) => {
    console.error(`[registry-verify] ${error.message}`);
    process.exit(1);
  });
}

async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  if (!parsed.model) {
    usage();
    process.exit(2);
  }
  if (!parsed.registryUrl) {
    throw new Error('Registry URL is empty. Set --registry-url or DOPPLER_HF_REGISTRY_URL.');
  }

  const registry = await loadRegistry(parsed.registryUrl);
  const entry = findRegistryEntry(registry, parsed.model);
  if (!entry) {
    const available = (registry.models || [])
      .map((model) => model?.modelId)
      .filter((id) => typeof id === 'string' && id.length > 0)
      .slice(0, 12);
    throw new Error(
      `Model alias "${parsed.model}" not found in registry. ` +
      `Try one of: ${available.join(', ')}`
    );
  }

  const modelUrl = resolveModelUrl(entry);
  const request = {
    suite: 'inference',
    modelId: entry.modelId,
    modelUrl,
    loadMode: 'http',
    cacheMode: 'warm',
    runtimePreset: 'modes/debug',
  };
  const run = { surface: parsed.surface || 'auto' };

  runVerify({ request, run }, parsed.surface);
}

main().catch((error) => {
  console.error(`[registry-verify] ${error.message}`);
  process.exit(1);
});
