#!/usr/bin/env node

import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DEFAULT_REGISTRY_URL = process.env.DOPPLER_HF_REGISTRY_URL
  || 'https://huggingface.co/Clocksmith/rdrr/resolve/main/registry/catalog.json';
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_CATALOG_FILE = path.join(REPO_ROOT, 'models', 'catalog.json');

function usage() {
  console.error(
    'Usage: node tools/run-registry-verify.js <model-alias-or-id> [--registry-url <url>] [--surface <auto|node|browser>] [--update-catalog] [--catalog-file <path>]'
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
    updateCatalog: false,
    catalogFile: DEFAULT_CATALOG_FILE,
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
    if (arg === '--update-catalog') {
      out.updateCatalog = true;
      continue;
    }
    if (arg === '--catalog-file') {
      out.catalogFile = argv[i + 1] ? String(argv[i + 1]).trim() : '';
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

function parseJsonPayload(stdout) {
  const raw = String(stdout ?? '').trim();
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function extractContractGateStatus(payload) {
  const result = payload?.result;
  const executionContractArtifact = result?.metrics?.executionContractArtifact ?? null;
  const executionV0GraphContractArtifact = result?.metrics?.executionV0GraphContractArtifact ?? null;
  const contractOk = executionContractArtifact?.ok !== false;
  const graphOk = executionV0GraphContractArtifact?.ok !== false;
  return {
    contractOk,
    graphOk,
    executionContractArtifact,
    executionV0GraphContractArtifact,
  };
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

  return new Promise((resolve, reject) => {
    const child = spawn('node', args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: process.env,
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      const text = String(chunk);
      stdout += text;
      process.stdout.write(text);
    });
    child.stderr.on('data', (chunk) => {
      const text = String(chunk);
      stderr += text;
      process.stderr.write(text);
    });
    child.on('exit', (code) => {
      resolve({
        exitCode: code ?? 1,
        stdout,
        stderr,
        payload: parseJsonPayload(stdout),
      });
    });
    child.on('error', (error) => {
      reject(error);
    });
  });
}

function toIsoDate(value = new Date()) {
  return value.toISOString().slice(0, 10);
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

async function updateCatalogTestStatus(options) {
  const {
    catalogFile,
    modelId,
    surface,
    suite,
    result,
    source = 'registry-verify',
    contractArtifacts = null,
  } = options;

  if (!catalogFile) {
    throw new Error('Catalog file path is empty.');
  }
  const raw = await fs.readFile(catalogFile, 'utf8');
  const payload = JSON.parse(raw);
  if (!isPlainObject(payload) || !Array.isArray(payload.models)) {
    throw new Error(`Catalog payload is invalid at ${catalogFile}.`);
  }

  const model = payload.models.find((entry) => entry?.modelId === modelId);
  if (!model) {
    throw new Error(`Model "${modelId}" was not found in catalog ${catalogFile}.`);
  }

  const lifecycle = isPlainObject(model.lifecycle) ? model.lifecycle : {};
  const status = isPlainObject(lifecycle.status) ? lifecycle.status : {};
  const tested = isPlainObject(lifecycle.tested) ? lifecycle.tested : {};

  model.lifecycle = {
    ...lifecycle,
    status: {
      ...status,
      tested: result === 'pass' ? 'verified' : 'failed',
    },
    tested: {
      ...tested,
      suite,
      surface,
      result,
      lastVerifiedAt: toIsoDate(),
      source,
      contracts: contractArtifacts && typeof contractArtifacts === 'object'
        ? contractArtifacts
        : tested.contracts ?? null,
    },
  };

  payload.updatedAt = toIsoDate();
  await fs.writeFile(catalogFile, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  console.error(
    `[registry-verify] updated catalog lifecycle for ${modelId} ` +
    `(tested=${model.lifecycle.status.tested}, result=${result})`
  );
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

  let verifyResult = {
    exitCode: 1,
    payload: null,
  };
  try {
    verifyResult = await runVerify({ request, run }, parsed.surface);
  } catch (error) {
    console.error(`[registry-verify] ${error.message}`);
    process.exit(1);
  }

  const gateStatus = extractContractGateStatus(verifyResult.payload);
  let exitCode = verifyResult.exitCode;
  if (exitCode === 0 && gateStatus.contractOk === false) {
    console.error('[registry-verify] execution contract gate failed.');
    exitCode = 1;
  }
  if (exitCode === 0 && gateStatus.graphOk === false) {
    console.error('[registry-verify] execution-v0 graph gate failed.');
    exitCode = 1;
  }

  if (parsed.updateCatalog) {
    const result = exitCode === 0 ? 'pass' : 'fail';
    await updateCatalogTestStatus({
      catalogFile: parsed.catalogFile,
      modelId: entry.modelId,
      surface: parsed.surface || 'auto',
      suite: request.suite,
      result,
      source: 'registry-verify',
      contractArtifacts: {
        executionContractOk: gateStatus.executionContractArtifact?.ok ?? null,
        executionV0GraphOk: gateStatus.executionV0GraphContractArtifact?.ok ?? null,
      },
    });
  }

  process.exit(exitCode);
}

main().catch((error) => {
  console.error(`[registry-verify] ${error.message}`);
  process.exit(1);
});
