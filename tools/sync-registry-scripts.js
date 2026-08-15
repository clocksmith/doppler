#!/usr/bin/env node

import fs from 'node:fs/promises';

const DEFAULT_PACKAGE_FILE = 'package.json';
const GENERATED_SCRIPT_PREFIX = 'verify:';
const GENERATED_SCRIPT_COMMAND_PREFIX = 'node tools/run-registry-verify.js ';

function parseArgs(argv) {
  const out = {
    packageFile: DEFAULT_PACKAGE_FILE,
    check: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--registry-url') {
      if (!argv[i + 1]) throw new Error('--registry-url requires a value.');
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

function buildNextScripts(currentScripts) {
  const base = removePreviouslyGeneratedScripts(currentScripts);
  base.verify = 'node tools/run-registry-verify.js';
  base['verify:model'] = 'node src/cli/doppler-cli.js verify';
  base['registry:sync:scripts'] = 'node tools/sync-registry-scripts.js';
  base['registry:sync:scripts:check'] = 'node tools/sync-registry-scripts.js --check';
  return base;
}

async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  const packageJson = await loadPackageJson(parsed.packageFile);
  const nextScripts = buildNextScripts(packageJson.scripts);
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
      '[registry-scripts] canonical verify commands are current (no per-model aliases)'
    );
    return;
  }

  packageJson.scripts = nextScripts;
  await fs.writeFile(parsed.packageFile, `${JSON.stringify(packageJson, null, 2)}\n`, 'utf8');
  console.log(
    `[registry-scripts] wrote canonical verify commands to ${parsed.packageFile}`
  );
}

main().catch((error) => {
  console.error(`[registry-scripts] ${error.message}`);
  process.exit(1);
});
