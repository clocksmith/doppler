#!/usr/bin/env node

/**
 * Migrate a v0 execution manifest to v1 format (manifest-only, no weight re-conversion).
 *
 * Usage:
 *   node tools/migrate-manifest-v0-to-v1.js <manifest-path> [--dry-run] [--verbose]
 *   node tools/migrate-manifest-v0-to-v1.js --hf <model-path> [--dry-run] [--verbose]
 *
 * Examples:
 *   node tools/migrate-manifest-v0-to-v1.js /media/x/models/rdrr/gemma-3-1b-it-q4k-ehf16-af32/manifest.json
 *   node tools/migrate-manifest-v0-to-v1.js --hf models/gemma-3-1b-it-q4k-ehf16-af32 --dry-run
 */

import fs from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { getKernelRefContentDigest } from '../src/config/kernels/kernel-ref.js';

const EXECUTION_V0_SCHEMA_ID = 'doppler.execution/v0';
const EXECUTION_V1_SCHEMA_ID = 'doppler.execution/v1';

function buildKernelKey(kernel, entry) {
  const base = kernel.replace(/\.wgsl$/i, '').replace(/[^a-z0-9_]/gi, '_');
  const ent = entry === 'main' ? '' : `_${entry.replace(/[^a-z0-9_]/gi, '_')}`;
  return `${base}${ent}`;
}

function migrateV0ToV1(manifest) {
  const inference = manifest.inference;
  if (!inference) throw new Error('manifest.inference is missing');
  if (inference.schema !== EXECUTION_V0_SCHEMA_ID) {
    throw new Error(`Expected schema "${EXECUTION_V0_SCHEMA_ID}", got "${inference.schema}"`);
  }

  const v0Steps = inference.execution?.steps;
  if (!Array.isArray(v0Steps) || v0Steps.length === 0) {
    throw new Error('manifest.inference.execution.steps is empty or missing');
  }

  // Step 1: Build kernel declarations from unique kernel+entry pairs
  const kernels = {};
  const kernelKeyMap = new Map(); // `kernel#entry` → kernelKey
  for (const step of v0Steps) {
    if (step.op === 'cast') continue;
    const kernel = step.kernel;
    const entry = step.entry ?? 'main';
    const mapKey = `${kernel}#${entry}`;
    if (kernelKeyMap.has(mapKey)) continue;

    const kernelKey = buildKernelKey(kernel, entry);
    let digest;
    try {
      const hex = getKernelRefContentDigest(kernel, entry);
      digest = `sha256:${hex}`;
    } catch {
      // Fall back to kernelRef digest if content digest lookup fails
      const ref = step.kernelRef;
      if (ref?.digest) {
        digest = ref.digest;
      } else {
        throw new Error(`No digest for kernel "${mapKey}" and no kernelRef.digest on step "${step.id}"`);
      }
    }
    kernels[kernelKey] = {
      kernel,
      entry,
      digest,
      ...(step.constants ? { constants: step.constants } : {}),
    };
    kernelKeyMap.set(mapKey, kernelKey);
  }

  // Step 2: Group steps by section and phase
  const preLayer = [];
  const postLayer = [];
  const decodeSteps = [];
  const prefillSteps = [];

  for (const step of v0Steps) {
    if (step.op === 'cast') continue;
    const kernel = step.kernel;
    const entry = step.entry ?? 'main';
    const kernelKey = kernelKeyMap.get(`${kernel}#${entry}`);
    if (!kernelKey) continue;

    const tuple = step.weights
      ? [step.op, kernelKey, step.weights]
      : [step.op, kernelKey];

    const section = step.section;
    const phase = step.phase;

    if (section === 'preLayer') {
      preLayer.push(tuple);
    } else if (section === 'postLayer' || section === 'sampling') {
      postLayer.push(tuple);
    } else if (section === 'layer') {
      if (phase === 'decode' || phase === 'both') {
        decodeSteps.push({ tuple, layers: step.layers });
      }
      if (phase === 'prefill' || phase === 'both') {
        prefillSteps.push({ tuple, layers: step.layers });
      }
    }
  }

  // Step 3: Build v1 step entries, grouping by layer targeting
  function buildStepEntries(steps) {
    const allLayerSteps = [];
    const layerGroups = new Map(); // JSON(layers) → { layers, steps }

    for (const { tuple, layers } of steps) {
      if (layers === 'all') {
        allLayerSteps.push(tuple);
      } else if (Array.isArray(layers)) {
        const key = JSON.stringify(layers);
        if (!layerGroups.has(key)) {
          layerGroups.set(key, { layers, steps: [] });
        }
        layerGroups.get(key).steps.push(tuple);
      }
    }

    const entries = [...allLayerSteps];
    for (const group of layerGroups.values()) {
      entries.push({ layers: group.layers, steps: group.steps });
    }
    return entries;
  }

  // Step 4: Build session defaults from v0
  const v0Session = inference.sessionDefaults;
  const sessionDefaults = {
    compute: {
      defaults: {
        activationDtype: v0Session?.compute?.defaults?.activationDtype ?? 'f16',
        mathDtype: v0Session?.compute?.defaults?.mathDtype ?? 'f16',
        accumDtype: v0Session?.compute?.defaults?.accumDtype ?? 'f32',
        outputDtype: v0Session?.compute?.defaults?.outputDtype ?? 'f16',
      },
    },
    kvcache: v0Session?.kvcache ?? null,
    decodeLoop: v0Session?.decodeLoop ?? null,
  };

  // Step 5: Build v1 execution graph
  const execution = {
    kernels,
    preLayer,
    decode: buildStepEntries(decodeSteps),
    prefill: buildStepEntries(prefillSteps),
    postLayer,
    policies: {
      unsupportedPrecision: 'error',
      dtypeTransition: 'require_cast_step',
      unresolvedKernel: 'error',
    },
  };

  // Step 6: Update manifest inference
  const updatedInference = { ...inference };
  updatedInference.schema = EXECUTION_V1_SCHEMA_ID;
  updatedInference.sessionDefaults = sessionDefaults;
  updatedInference.execution = execution;
  // Remove v0-only fields
  delete updatedInference.executionPatch;

  return { ...manifest, inference: updatedInference };
}

function injectV1FromConfig(manifest, configPath) {
  const raw = JSON.parse(readFileSync(configPath, 'utf8'));
  const execution = raw.execution;
  if (!execution || !execution.kernels) {
    throw new Error(`Config ${configPath} does not have a v1 execution graph`);
  }
  const sessionDefaults = raw.sessionDefaults ?? {
    compute: { defaults: { activationDtype: 'f16', mathDtype: 'f16', accumDtype: 'f32', outputDtype: 'f16' } },
    kvcache: null,
    decodeLoop: null,
  };
  const updatedInference = { ...manifest.inference };
  updatedInference.schema = EXECUTION_V1_SCHEMA_ID;
  updatedInference.execution = execution;
  updatedInference.sessionDefaults = sessionDefaults;
  delete updatedInference.executionPatch;
  return { ...manifest, inference: updatedInference };
}

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const verbose = args.includes('--verbose');
  const hfIndex = args.indexOf('--hf');
  const configIndex = args.indexOf('--config');
  const filtered = args.filter((a) => !a.startsWith('--'));

  let manifestPath;
  if (hfIndex !== -1 && args[hfIndex + 1]) {
    // HF mode: fetch manifest from Clocksmith/rdrr
    const hfPath = args[hfIndex + 1];
    const configPath = configIndex !== -1 ? args[configIndex + 1] : null;
    const url = `https://huggingface.co/Clocksmith/rdrr/resolve/main/${hfPath}/manifest.json`;
    console.log(`Fetching: ${url}`);
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
    const manifest = await resp.json();

    let migrated;
    if (configPath) {
      // Inject v1 execution from conversion config
      migrated = injectV1FromConfig(manifest, configPath);
    } else if (manifest.inference?.schema === EXECUTION_V0_SCHEMA_ID) {
      migrated = migrateV0ToV1(manifest);
    } else {
      throw new Error(
        `Manifest has schema="${manifest.inference?.schema}" — not v0. ` +
        'Use --config <path> to inject v1 execution from a conversion config.'
      );
    }

    const output = JSON.stringify(migrated, null, 2) + '\n';
    if (verbose) console.log(output);
    if (dryRun) {
      console.log(`[dry-run] Would write ${output.length} bytes`);
      const stepCount = migrated.inference.execution.decode.length +
        migrated.inference.execution.prefill.length +
        migrated.inference.execution.preLayer.length +
        migrated.inference.execution.postLayer.length;
      const kernelCount = Object.keys(migrated.inference.execution.kernels).length;
      console.log(`[dry-run] ${kernelCount} kernels, ${stepCount} steps`);
    } else {
      console.log('[hf] Manifest migrated to stdout. Redirect to file or use local path mode to write.');
      process.stdout.write(output);
    }
    return;
  }

  manifestPath = filtered[0];
  if (!manifestPath) {
    console.error('Usage: node tools/migrate-manifest-v0-to-v1.js <manifest-path> [--dry-run] [--verbose]');
    process.exit(1);
  }

  manifestPath = path.resolve(manifestPath);
  const raw = await fs.readFile(manifestPath, 'utf8');
  const manifest = JSON.parse(raw);

  if (manifest.inference?.schema === EXECUTION_V1_SCHEMA_ID) {
    console.log(`Already v1: ${manifestPath}`);
    return;
  }

  if (manifest.inference?.schema !== EXECUTION_V0_SCHEMA_ID) {
    console.error(`Unsupported schema: ${manifest.inference?.schema} (expected ${EXECUTION_V0_SCHEMA_ID})`);
    process.exit(1);
  }

  const migrated = migrateV0ToV1(manifest);
  const output = JSON.stringify(migrated, null, 2) + '\n';

  const stepCount = migrated.inference.execution.decode.length +
    migrated.inference.execution.prefill.length +
    migrated.inference.execution.preLayer.length +
    migrated.inference.execution.postLayer.length;
  const kernelCount = Object.keys(migrated.inference.execution.kernels).length;

  if (verbose) {
    console.log(output);
  }

  if (dryRun) {
    console.log(`[dry-run] ${manifestPath}`);
    console.log(`  schema: ${EXECUTION_V0_SCHEMA_ID} → ${EXECUTION_V1_SCHEMA_ID}`);
    console.log(`  kernels: ${kernelCount}`);
    console.log(`  steps: ${stepCount}`);
    console.log(`  size: ${raw.length} → ${output.length} bytes`);
  } else {
    await fs.writeFile(manifestPath, output, 'utf8');
    console.log(`Migrated: ${manifestPath}`);
    console.log(`  schema: ${EXECUTION_V1_SCHEMA_ID}`);
    console.log(`  kernels: ${kernelCount}, steps: ${stepCount}`);
  }
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
