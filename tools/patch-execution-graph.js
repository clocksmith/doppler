#!/usr/bin/env node

// patch-execution-graph.js — Update execution graphs and inference behavior in
// RDRR manifests from canonical conversion configs without reconverting.
//
// Syncs the following from the conversion config to the manifest:
//   - inference.execution (kernels + step tuples)
//   - inference.sessionDefaults (compute, kvcache, decodeLoop)
//   - inference.normalization (rmsNormWeightOffset, rmsNormEps, etc.)
//   - inference.attention (queryPreAttnScalar, queryKeyNorm, etc.)
//   - inference.ffn (activation, gatedActivation, etc.)
//   - inference.rope (ropeTheta, partialRotaryFactor, etc.)
//   - inference.output (finalLogitSoftcapping, tieWordEmbeddings, etc.)
//
// Everything else (shards, tensors, architecture, quantization, etc.) is
// preserved.
//
// Usage:
//   node tools/patch-execution-graph.js <rdrr-root> [options]
//
// Options:
//   --config-dir <dir>   Conversion config directory (default: src/config/conversion)
//   --model <id>         Patch only this model (by directory name or modelId)
//   --dry-run            Show what would change without writing
//   --force              Overwrite even if execution graph is identical
//   --verbose            Print detailed diff information

import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const DEFAULT_CONFIG_DIR = path.resolve(
  new URL('.', import.meta.url).pathname,
  '../src/config/conversion'
);

function fail(message) {
  console.error(`[patch-execution-graph] ${message}`);
  process.exit(1);
}

function parseArgs(argv) {
  const args = {
    rdrrRoot: null,
    configDir: DEFAULT_CONFIG_DIR,
    model: null,
    dryRun: false,
    force: false,
    verbose: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--config-dir') {
      args.configDir = path.resolve(argv[i + 1] ?? '');
      i += 1;
      continue;
    }
    if (arg === '--model') {
      args.model = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--dry-run') {
      args.dryRun = true;
      continue;
    }
    if (arg === '--force') {
      args.force = true;
      continue;
    }
    if (arg === '--verbose') {
      args.verbose = true;
      continue;
    }
    if (arg.startsWith('-')) {
      fail(`Unknown flag: ${arg}`);
    }
    if (!args.rdrrRoot) {
      args.rdrrRoot = path.resolve(arg);
      continue;
    }
    fail(`Unexpected positional argument: ${arg}`);
  }

  if (!args.rdrrRoot) {
    fail(
      'Usage: node tools/patch-execution-graph.js <rdrr-root> [--config-dir <dir>] [--model <id>] [--dry-run] [--force] [--verbose]'
    );
  }

  return args;
}

async function readJson(filePath) {
  const raw = await fs.readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

async function readJsonSafe(filePath) {
  try {
    return await readJson(filePath);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Build conversion config index: modelBaseId → config path + parsed config
// ---------------------------------------------------------------------------

async function collectJsonFiles(dir) {
  const results = [];
  const stack = [dir];
  while (stack.length > 0) {
    const current = stack.pop();
    let entries;
    try {
      entries = await fs.readdir(current, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.json')) {
        results.push(fullPath);
      }
    }
  }
  return results.sort();
}

async function buildConfigIndex(configDir) {
  const configFiles = await collectJsonFiles(configDir);
  const index = new Map();

  for (const configPath of configFiles) {
    const config = await readJsonSafe(configPath);
    if (!config || typeof config !== 'object') continue;

    const modelBaseId =
      config?.output?.modelBaseId ??
      path.basename(configPath, '.json');

    if (!config.execution && !config.inference?.execution) continue;

    index.set(modelBaseId, { configPath, config });
  }

  return index;
}

// ---------------------------------------------------------------------------
// Matching: find conversion config for a given RDRR model
// ---------------------------------------------------------------------------

function normalizeId(id) {
  return id
    .toLowerCase()
    .replace(/[._]/g, '-');
}

function findConfigForModel(configIndex, modelId, dirName) {
  // Exact match on modelBaseId
  if (configIndex.has(modelId)) {
    return configIndex.get(modelId);
  }
  if (configIndex.has(dirName)) {
    return configIndex.get(dirName);
  }

  // Normalized match
  const normalizedModelId = normalizeId(modelId);
  const normalizedDirName = normalizeId(dirName);

  for (const [baseId, entry] of configIndex) {
    const normalizedBase = normalizeId(baseId);
    if (
      normalizedBase === normalizedModelId ||
      normalizedBase === normalizedDirName
    ) {
      return entry;
    }
  }

  // Substring containment — try if one contains the other
  for (const [baseId, entry] of configIndex) {
    const normalizedBase = normalizeId(baseId);
    if (
      normalizedModelId.includes(normalizedBase) ||
      normalizedBase.includes(normalizedModelId)
    ) {
      return entry;
    }
  }

  return null;
}

// ---------------------------------------------------------------------------
// Inference behavior blocks synced from conversion config → manifest.
// These live at inference.* in both the conversion config and the manifest.
// ---------------------------------------------------------------------------

const INFERENCE_BEHAVIOR_KEYS = [
  'normalization',
  'attention',
  'ffn',
  'rope',
  'output',
];

// ---------------------------------------------------------------------------
// Extract execution + sessionDefaults + behavior blocks from conversion config
// ---------------------------------------------------------------------------

function extractFromConfig(config) {
  // execution and sessionDefaults may be top-level or nested under inference
  const execution = config.execution ?? config.inference?.execution ?? null;
  const sessionDefaults =
    config.sessionDefaults ?? config.inference?.sessionDefaults ?? null;

  if (!execution || typeof execution.kernels !== 'object') {
    return null;
  }

  // Inference behavior blocks always live under config.inference.*
  const behavior = {};
  for (const key of INFERENCE_BEHAVIOR_KEYS) {
    const block = config.inference?.[key] ?? null;
    if (block) {
      behavior[key] = block;
    }
  }

  return { execution, sessionDefaults, behavior };
}

// ---------------------------------------------------------------------------
// Diff reporting
// ---------------------------------------------------------------------------

function summarizeKernelDiff(oldKernels, newKernels) {
  const changes = [];
  const oldKeys = new Set(Object.keys(oldKernels ?? {}));
  const newKeys = new Set(Object.keys(newKernels ?? {}));

  for (const key of newKeys) {
    if (!oldKeys.has(key)) {
      changes.push(`  + kernel "${key}": ${newKernels[key].kernel}`);
    } else {
      const oldEntry = oldKernels[key];
      const newEntry = newKernels[key];
      if (
        oldEntry.kernel !== newEntry.kernel ||
        oldEntry.entry !== newEntry.entry
      ) {
        changes.push(
          `  ~ kernel "${key}": ${oldEntry.kernel}:${oldEntry.entry} -> ${newEntry.kernel}:${newEntry.entry}`
        );
      } else if (oldEntry.digest !== newEntry.digest) {
        changes.push(`  ~ kernel "${key}": digest updated`);
      }
    }
  }

  for (const key of oldKeys) {
    if (!newKeys.has(key)) {
      changes.push(`  - kernel "${key}": removed`);
    }
  }

  return changes;
}

function summarizeStepDiff(phase, oldSteps, newSteps) {
  const oldStr = JSON.stringify(oldSteps ?? []);
  const newStr = JSON.stringify(newSteps ?? []);
  if (oldStr === newStr) return [];
  const oldCount = (oldSteps ?? []).length;
  const newCount = (newSteps ?? []).length;
  return [`  ~ ${phase}: ${oldCount} steps -> ${newCount} steps`];
}

function summarizeBehaviorDiff(blockName, oldBlock, newBlock) {
  const changes = [];
  const allKeys = new Set([
    ...Object.keys(oldBlock ?? {}),
    ...Object.keys(newBlock ?? {}),
  ]);
  for (const key of allKeys) {
    const oldVal = oldBlock?.[key];
    const newVal = newBlock?.[key];
    if (JSON.stringify(oldVal) !== JSON.stringify(newVal)) {
      changes.push(`  ~ ${blockName}.${key}: ${JSON.stringify(oldVal)} -> ${JSON.stringify(newVal)}`);
    }
  }
  if (changes.length === 0) {
    changes.push(`  ~ ${blockName} updated`);
  }
  return changes;
}

// ---------------------------------------------------------------------------
// Patch a single manifest
// ---------------------------------------------------------------------------

async function patchManifest(manifestPath, configEntry, args) {
  const manifest = await readJson(manifestPath);
  const modelId = manifest.modelId ?? path.basename(path.dirname(manifestPath));

  const extracted = extractFromConfig(configEntry.config);
  if (!extracted) {
    return { modelId, status: 'skip', reason: 'no execution block in config' };
  }

  const oldExecution = manifest.inference?.execution ?? null;
  const oldSessionDefaults = manifest.inference?.sessionDefaults ?? null;
  const newExecution = extracted.execution;
  const newSessionDefaults = extracted.sessionDefaults;

  // Check execution and sessionDefaults
  const executionMatch =
    JSON.stringify(oldExecution) === JSON.stringify(newExecution);
  const sessionMatch =
    JSON.stringify(oldSessionDefaults) === JSON.stringify(newSessionDefaults);

  // Check inference behavior blocks
  const behaviorDiffs = [];
  for (const key of INFERENCE_BEHAVIOR_KEYS) {
    const newBlock = extracted.behavior[key];
    if (!newBlock) continue;
    const oldBlock = manifest.inference?.[key] ?? null;
    if (JSON.stringify(oldBlock) !== JSON.stringify(newBlock)) {
      behaviorDiffs.push(key);
    }
  }

  if (executionMatch && sessionMatch && behaviorDiffs.length === 0 && !args.force) {
    return { modelId, status: 'unchanged' };
  }

  // Build diff summary
  const changes = [];
  if (!executionMatch) {
    changes.push(
      ...summarizeKernelDiff(oldExecution?.kernels, newExecution?.kernels)
    );
    for (const phase of ['preLayer', 'decode', 'prefill', 'postLayer']) {
      changes.push(
        ...summarizeStepDiff(phase, oldExecution?.[phase], newExecution?.[phase])
      );
    }
  }
  if (!sessionMatch) {
    changes.push('  ~ sessionDefaults updated');
  }
  for (const key of behaviorDiffs) {
    const oldBlock = manifest.inference?.[key] ?? {};
    const newBlock = extracted.behavior[key];
    const fieldChanges = summarizeBehaviorDiff(key, oldBlock, newBlock);
    changes.push(...fieldChanges);
  }

  if (args.dryRun) {
    return { modelId, status: 'would-patch', changes, configPath: configEntry.configPath };
  }

  // Patch in place
  if (!manifest.inference || typeof manifest.inference !== 'object') {
    manifest.inference = {};
  }
  manifest.inference.execution = newExecution;
  manifest.inference.sessionDefaults = newSessionDefaults;

  // Sync behavior blocks
  for (const key of INFERENCE_BEHAVIOR_KEYS) {
    const newBlock = extracted.behavior[key];
    if (newBlock) {
      manifest.inference[key] = newBlock;
    }
  }

  // Stamp patch metadata
  if (!manifest.metadata || typeof manifest.metadata !== 'object') {
    manifest.metadata = {};
  }
  manifest.metadata.executionGraphPatch = {
    at: new Date().toISOString(),
    config: path.basename(configEntry.configPath),
  };

  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2), 'utf8');

  return { modelId, status: 'patched', changes, configPath: configEntry.configPath };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const configIndex = await buildConfigIndex(args.configDir);
  console.error(
    `[patch-execution-graph] loaded ${configIndex.size} conversion configs from ${args.configDir}`
  );

  // Collect model directories
  let modelDirs;
  try {
    const entries = await fs.readdir(args.rdrrRoot, { withFileTypes: true });
    modelDirs = entries
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
      .sort();
  } catch (error) {
    fail(`Cannot read RDRR root ${args.rdrrRoot}: ${error.message}`);
  }

  if (args.model) {
    modelDirs = modelDirs.filter(
      (d) => d === args.model || normalizeId(d) === normalizeId(args.model)
    );
    if (modelDirs.length === 0) {
      fail(`Model "${args.model}" not found in ${args.rdrrRoot}`);
    }
  }

  const results = [];

  for (const dirName of modelDirs) {
    const modelDir = path.join(args.rdrrRoot, dirName);
    const manifestPath = path.join(modelDir, 'manifest.json');

    const manifest = await readJsonSafe(manifestPath);
    if (!manifest) {
      results.push({ modelId: dirName, status: 'skip', reason: 'no manifest.json' });
      continue;
    }

    const modelId = manifest.modelId ?? dirName;
    const configEntry = findConfigForModel(configIndex, modelId, dirName);

    if (!configEntry) {
      results.push({ modelId, status: 'skip', reason: 'no matching conversion config' });
      continue;
    }

    const result = await patchManifest(manifestPath, configEntry, args);
    results.push(result);
  }

  // Report
  console.error('');
  const patched = results.filter((r) => r.status === 'patched' || r.status === 'would-patch');
  const unchanged = results.filter((r) => r.status === 'unchanged');
  const skipped = results.filter((r) => r.status === 'skip');

  for (const r of patched) {
    const verb = r.status === 'would-patch' ? 'would patch' : 'patched';
    console.log(
      `${verb}: ${r.modelId} (from ${path.basename(r.configPath)})`
    );
    if (args.verbose && r.changes?.length > 0) {
      for (const c of r.changes) {
        console.log(c);
      }
    }
  }

  for (const r of unchanged) {
    console.log(`unchanged: ${r.modelId}`);
  }

  for (const r of skipped) {
    console.log(`skip: ${r.modelId} (${r.reason})`);
  }

  console.error('');
  console.error(
    `[patch-execution-graph] ${patched.length} patched, ${unchanged.length} unchanged, ${skipped.length} skipped`
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    fail(error?.message || String(error));
  });
}

export { buildConfigIndex, findConfigForModel, extractFromConfig };
