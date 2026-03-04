#!/usr/bin/env node

import fs from 'node:fs/promises';

const DEFAULT_REGISTRY_URL = process.env.DOPPLER_HF_REGISTRY_URL
  || 'https://huggingface.co/Clocksmith/rdrr/resolve/main/registry/catalog.json';
const DEFAULT_PACKAGE_FILE = 'package.json';
const GENERATED_SCRIPT_PREFIX = 'verify:';
const GENERATED_SCRIPT_COMMAND_PREFIX = 'node tools/run-registry-verify.js ';

function parseArgs(argv) {
  const out = {
    registryUrl: DEFAULT_REGISTRY_URL,
    packageFile: DEFAULT_PACKAGE_FILE,
    check: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--registry-url') {
      out.registryUrl = argv[i + 1] ? String(argv[i + 1]).trim() : '';
      i += 1;
      continue;
    }
    if (arg === '--package-file') {
      out.packageFile = argv[i + 1] ? String(argv[i + 1]).trim() : '';
      i += 1;
      continue;
    }
    if (arg === '--check') {
      out.check = true;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  return out;
}

function normalizeToken(value) {
  return typeof value === 'string' ? value.trim().toLowerCase() : '';
}

function normalizeAliasForScript(value) {
  const raw = normalizeToken(value);
  if (!raw) return null;
  const collapsed = raw
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+/, '')
    .replace(/-+$/, '');
  if (!collapsed) return null;
  return collapsed;
}

function collectModelTokens(model) {
  const tokens = [];
  if (typeof model?.modelId === 'string') {
    tokens.push(model.modelId);
  }
  const aliases = Array.isArray(model?.aliases) ? model.aliases : [];
  for (const alias of aliases) {
    if (typeof alias === 'string' && alias.trim()) {
      tokens.push(alias);
    }
  }
  return tokens;
}

function buildGeneratedVerifyScripts(registry) {
  const models = Array.isArray(registry?.models) ? registry.models : [];
  const scriptMap = new Map();
  const conflictMap = new Map();

  for (const model of models) {
    if (!model || typeof model !== 'object') continue;
    const modelId = typeof model.modelId === 'string' ? model.modelId.trim() : '';
    if (!modelId) continue;

    const seenTokens = new Set();
    for (const token of collectModelTokens(model)) {
      const alias = normalizeAliasForScript(token);
      if (!alias || seenTokens.has(alias)) continue;
      seenTokens.add(alias);

      const scriptName = `${GENERATED_SCRIPT_PREFIX}${alias}`;
      const scriptValue = `${GENERATED_SCRIPT_COMMAND_PREFIX}${alias}`;

      if (scriptMap.has(scriptName) && scriptMap.get(scriptName) !== scriptValue) {
        conflictMap.set(scriptName, [scriptMap.get(scriptName), scriptValue]);
        continue;
      }
      scriptMap.set(scriptName, scriptValue);
    }
  }

  if (conflictMap.size > 0) {
    const examples = [...conflictMap.keys()].slice(0, 5).join(', ');
    throw new Error(`Registry alias collisions detected for scripts: ${examples}`);
  }

  return Object.fromEntries(
    [...scriptMap.entries()].sort((a, b) => a[0].localeCompare(b[0]))
  );
}

function removePreviouslyGeneratedScripts(scripts) {
  const next = {};
  for (const [key, value] of Object.entries(scripts || {})) {
    if (
      key.startsWith(GENERATED_SCRIPT_PREFIX)
      && typeof value === 'string'
      && value.startsWith(GENERATED_SCRIPT_COMMAND_PREFIX)
    ) {
      continue;
    }
    next[key] = value;
  }
  return next;
}

async function loadRegistry(registryUrl) {
  if (!registryUrl) {
    throw new Error('Registry URL is required.');
  }
  const response = await fetch(registryUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch registry ${registryUrl}: HTTP ${response.status}.`);
  }
  const payload = await response.json();
  if (!payload || typeof payload !== 'object' || !Array.isArray(payload.models)) {
    throw new Error(`Invalid registry payload at ${registryUrl}.`);
  }
  return payload;
}

async function loadPackageJson(packageFile) {
  const raw = await fs.readFile(packageFile, 'utf8');
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${packageFile} must be a JSON object.`);
  }
  if (!parsed.scripts || typeof parsed.scripts !== 'object' || Array.isArray(parsed.scripts)) {
    parsed.scripts = {};
  }
  return parsed;
}

function buildNextScripts(currentScripts, generatedScripts) {
  const base = removePreviouslyGeneratedScripts(currentScripts);
  base.verify = 'node tools/run-registry-verify.js';
  base['verify:model'] = 'node tools/doppler-cli.js verify';
  base['registry:sync:scripts'] = 'node tools/sync-registry-scripts.js';
  base['registry:sync:scripts:check'] = 'node tools/sync-registry-scripts.js --check';

  for (const [key, value] of Object.entries(generatedScripts)) {
    if (key === 'verify' || key === 'verify:model') continue;
    base[key] = value;
  }
  return base;
}

async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  const registry = await loadRegistry(parsed.registryUrl);
  const packageJson = await loadPackageJson(parsed.packageFile);
  const generatedScripts = buildGeneratedVerifyScripts(registry);

  const nextScripts = buildNextScripts(packageJson.scripts, generatedScripts);
  const currentSerialized = JSON.stringify(packageJson.scripts);
  const nextSerialized = JSON.stringify(nextScripts);

  if (parsed.check) {
    if (currentSerialized !== nextSerialized) {
      throw new Error(
        `Registry scripts are out of date in ${parsed.packageFile}. ` +
        `Run: node tools/sync-registry-scripts.js`
      );
    }
    console.log(
      `[registry-scripts] up to date (${Object.keys(generatedScripts).length} generated verify aliases)`
    );
    return;
  }

  packageJson.scripts = nextScripts;
  await fs.writeFile(parsed.packageFile, `${JSON.stringify(packageJson, null, 2)}\n`, 'utf8');
  console.log(
    `[registry-scripts] wrote ${Object.keys(generatedScripts).length} generated verify aliases to ${parsed.packageFile}`
  );
}

main().catch((error) => {
  console.error(`[registry-scripts] ${error.message}`);
  process.exit(1);
});
