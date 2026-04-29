#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { TRANSFORMS } from '../src/config/transforms/execution-graph-transforms.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

const DEFAULT_CATALOG_PATH = 'models/catalog.json';
const DEFAULT_MANIFEST_ROOT = 'models/local';
const DEFAULT_REGISTRY_PATH = 'src/config/kernels/registry.json';
const DEFAULT_CAPABILITY_RULES_PATH = 'src/rules/inference/capability-transforms.rules.json';
const DEFAULT_USAGE_ALLOWLIST_PATH = 'tools/policies/kernel-usage-allowlist.json';
const DEFAULT_LIMIT = 80;
const REPO_CONTRACT_REFERENCE_ROOTS = [
  'src/config/conversion',
  'src/config/source-packages',
  'src/config/transforms',
  'src/gpu/kernels/codegen',
  'src/rules',
  'tests',
  'demo/data',
];
const JS_DISPATCH_REFERENCE_ROOTS = [
  'src/gpu',
  'src/inference',
  'src/generation',
  'src/experimental',
  'demo',
];
const TEXT_REFERENCE_EXTENSIONS = new Set(['.d.ts', '.html', '.js', '.json']);
const REFERENCE_SCAN_EXCLUDED_SEGMENTS = new Set([
  'node_modules',
  '.git',
  'tests/fixtures/reports',
  'tools/data',
]);
const SESSION_ONLY_TRANSFORMS = new Set(['disableRetainQ4KMaterialization']);
const REACHABILITY_PROTECTED_STATUSES = new Set(['pinned', 'model-selectable', 'selectable']);
const ALLOWLIST_MATCH_FIELDS = new Set([
  'keys',
  'operations',
  'operationPrefixes',
  'operationSuffixes',
  'variants',
  'variantPrefixes',
  'variantSuffixes',
  'wgsl',
  'wgslPrefixes',
  'wgslSuffixes',
  'reachabilityStatuses',
]);
const CAPABILITY_KEYS = new Set(['hasSubgroups', 'hasF16', 'hasSubgroupsF16', 'maxWorkgroupStorageSize']);
const PLATFORM_KEY_MAP = new Map([
  ['platformId', 'id'],
  ['platformVendor', 'vendor'],
  ['platformArchitecture', 'architecture'],
]);
const RUNTIME_CONTEXT_KEYS = new Set([
  'activationDtype',
  'mathDtype',
  'accumDtype',
  'kvDtype',
  'retainQ4KMaterialization',
]);
const GRAPH_CONTEXT_KEYS = new Set([
  'headDim',
  'requiresF16ActivationNarrowing',
  'hasDensePrefillProjectionKernel',
  'hasQ4DecodeProjectionKernel',
  'hasQ4PrefillProjectionKernel',
  'hasAvailableQ4PrefillProjectionKernel',
]);
const PROJECTION_OPS = new Set([
  'q_proj',
  'k_proj',
  'v_proj',
  'o_proj',
  'gate_proj',
  'up_proj',
  'down_proj',
]);
const DENSE_Q4_PREFILL_FILES = new Set([
  'matmul_f16w_f32a.wgsl',
  'matmul_f16w_f32a_tiled.wgsl',
  'matmul_f16.wgsl',
  'matmul_f16_tiled.wgsl',
]);
const F32_ACTIVATION_NARROWING_FILES = new Set([
  'rmsnorm.wgsl',
  'rope.wgsl',
  'residual.wgsl',
  'gelu.wgsl',
  'sample.wgsl',
  'gather.wgsl',
  'matmul_gemv_subgroup.wgsl',
  'matmul_f16w_f32a.wgsl',
  'matmul_f16w_f32a_tiled.wgsl',
  'attention_decode_online_f16kv.wgsl',
  'attention_decode_chunked_f16kv.wgsl',
  'attention_small_f16kv.wgsl',
  'attention_streaming_f16kv.wgsl',
  'attention_decode.wgsl',
  'attention_small.wgsl',
  'attention_streaming.wgsl',
  'silu.wgsl',
]);

function parseArgs(argv) {
  const options = {
    json: false,
    check: false,
    includeUntested: false,
    allCatalog: false,
    manifestOnly: false,
    includeRuntimeTransformVariants: false,
    noAllowlist: false,
    failOnUnusedCandidates: false,
    failOnDeleteCandidates: false,
    modelIds: [],
    catalogPath: DEFAULT_CATALOG_PATH,
    manifestRoot: DEFAULT_MANIFEST_ROOT,
    registryPath: DEFAULT_REGISTRY_PATH,
    capabilityRulesPath: DEFAULT_CAPABILITY_RULES_PATH,
    usageAllowlistPath: DEFAULT_USAGE_ALLOWLIST_PATH,
    limit: DEFAULT_LIMIT,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--json') {
      options.json = true;
    } else if (arg === '--check') {
      options.check = true;
    } else if (arg === '--include-untested') {
      options.includeUntested = true;
    } else if (arg === '--all-catalog') {
      options.allCatalog = true;
    } else if (arg === '--manifest-only') {
      options.manifestOnly = true;
    } else if (arg === '--include-runtime-transform-variants') {
      options.includeRuntimeTransformVariants = true;
    } else if (arg === '--no-allowlist') {
      options.noAllowlist = true;
    } else if (arg === '--fail-on-unused-candidates') {
      options.failOnUnusedCandidates = true;
      options.failOnDeleteCandidates = true;
    } else if (arg === '--fail-on-delete-candidates') {
      options.failOnDeleteCandidates = true;
    } else if (arg === '--model') {
      options.modelIds.push(...readRequiredValue(argv, ++i, arg).split(',').filter(Boolean));
    } else if (arg === '--catalog') {
      options.catalogPath = readRequiredValue(argv, ++i, arg);
    } else if (arg === '--manifest-root') {
      options.manifestRoot = readRequiredValue(argv, ++i, arg);
    } else if (arg === '--registry') {
      options.registryPath = readRequiredValue(argv, ++i, arg);
    } else if (arg === '--capability-rules') {
      options.capabilityRulesPath = readRequiredValue(argv, ++i, arg);
    } else if (arg === '--allowlist') {
      options.usageAllowlistPath = readRequiredValue(argv, ++i, arg);
    } else if (arg === '--limit') {
      const value = Number.parseInt(readRequiredValue(argv, ++i, arg), 10);
      if (!Number.isFinite(value) || value < 0) {
        throw new Error('--limit must be a non-negative integer');
      }
      options.limit = value;
    } else if (arg === '--help' || arg === '-h') {
      options.help = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function readRequiredValue(argv, index, flag) {
  const value = argv[index];
  if (typeof value !== 'string' || value.startsWith('--')) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
}

function resolveRepoPath(value) {
  return path.isAbsolute(value) ? value : path.join(repoRoot, value);
}

function printHelp() {
  console.log(`Usage:
  npm run kernels:supported-manifests:report
  node tools/list-supported-manifest-kernels.js [options]

Options:
  --json                 Print full JSON report.
  --check                Exit non-zero for missing selected manifests or live kernel refs absent from the registry.
  --include-untested     Include active/ready catalog entries whose tested status is not verified.
  --all-catalog          Include every catalog entry, regardless of lifecycle status.
  --manifest-only        Do not count capability-transform fallback kernels as live.
  --include-runtime-transform-variants Include runtime-requested capability lanes, such as f16 activation profiles.
  --no-allowlist         Do not protect unused variants through the usage allowlist.
  --fail-on-delete-candidates Exit non-zero when delete candidates are present.
  --fail-on-unused-candidates Backward-compatible alias for --fail-on-delete-candidates.
  --model <id[,id]>      Restrict the selected catalog entries by modelId. Repeatable.
  --catalog <path>       Catalog JSON path. Defaults to ${DEFAULT_CATALOG_PATH}.
  --manifest-root <dir>  Directory containing <modelId>/manifest.json. Defaults to ${DEFAULT_MANIFEST_ROOT}.
  --registry <path>      Kernel registry JSON path. Defaults to ${DEFAULT_REGISTRY_PATH}.
  --capability-rules <path> Capability transform rules path. Defaults to ${DEFAULT_CAPABILITY_RULES_PATH}.
  --allowlist <path>     Kernel usage allowlist path. Defaults to ${DEFAULT_USAGE_ALLOWLIST_PATH}.
  --limit <n>            Human-output list limit. Defaults to ${DEFAULT_LIMIT}; 0 hides lists.`);
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function shouldSkipReferencePath(relativePath) {
  return [...REFERENCE_SCAN_EXCLUDED_SEGMENTS].some((segment) => relativePath === segment || relativePath.startsWith(`${segment}/`));
}

async function listTextReferenceFiles(rootPath, extensions = TEXT_REFERENCE_EXTENSIONS) {
  const files = [];
  async function visit(dirPath) {
    let entries;
    try {
      entries = await fs.readdir(dirPath, { withFileTypes: true });
    } catch (error) {
      if (error?.code === 'ENOENT') return;
      throw error;
    }

    for (const entry of entries) {
      const filePath = path.join(dirPath, entry.name);
      const relativePath = path.relative(repoRoot, filePath);
      if (shouldSkipReferencePath(relativePath)) continue;
      if (entry.isDirectory()) {
        await visit(filePath);
        continue;
      }
      if (!entry.isFile()) continue;
      if (hasTextReferenceExtension(entry.name, extensions)) {
        files.push(filePath);
      }
    }
  }

  await visit(rootPath);
  return files;
}

function hasTextReferenceExtension(fileName, extensions) {
  if (extensions.has(path.extname(fileName))) return true;
  return extensions.has('.d.ts') && fileName.endsWith('.d.ts');
}

async function collectRepoReferenceIndex(registryEntries) {
  const contractFiles = await collectReferenceFiles(REPO_CONTRACT_REFERENCE_ROOTS, TEXT_REFERENCE_EXTENSIONS);
  const jsDispatchFiles = await collectReferenceFiles(JS_DISPATCH_REFERENCE_ROOTS, new Set(['.js']));
  const byWgsl = new Map();
  const byEntryId = new Map();

  for (const filePath of contractFiles) {
    const text = await fs.readFile(filePath, 'utf8');
    const source = path.relative(repoRoot, filePath);
    for (const entry of registryEntries) {
      if (text.includes(entry.wgsl)) {
        pushSetMap(byWgsl, entry.wgsl, source);
      }
    }
  }

  for (const filePath of jsDispatchFiles) {
    const text = await fs.readFile(filePath, 'utf8');
    const source = path.relative(repoRoot, filePath);
    for (const pair of collectStaticDispatchPairs(text)) {
      pushSetMap(byEntryId, registryEntryId(pair), source);
    }
  }

  return {
    byWgsl,
    byEntryId,
    scannedFiles: {
      repoContract: contractFiles.length,
      jsDispatch: jsDispatchFiles.length,
    },
    roots: {
      repoContract: REPO_CONTRACT_REFERENCE_ROOTS,
      jsDispatch: JS_DISPATCH_REFERENCE_ROOTS,
    },
  };
}

async function collectReferenceFiles(roots, extensions) {
  const files = [];
  for (const root of roots) {
    files.push(...await listTextReferenceFiles(resolveRepoPath(root), extensions));
  }
  return [...new Set(files)].sort((a, b) => a.localeCompare(b));
}

function collectStaticDispatchPairs(text) {
  const pairs = [];
  collectTupleMatches(pairs, text, /(?:getPipelineFast|createPipeline|getKernelConfig|getPipelineFor)\(\s*['"]([^'"]+)['"]\s*,\s*['"]([^'"]+)['"]/g);
  collectTupleMatches(pairs, text, /unifiedKernelWrapper\(\s*['"]([^'"]+)['"]\s*,\s*[^,]+,\s*['"]([^'"]+)['"]/g);
  return pairs;
}

function collectTupleMatches(pairs, text, pattern) {
  for (const match of text.matchAll(pattern)) {
    pairs.push({ operation: match[1], variant: match[2] });
  }
}

function isVerifiedSupportedModel(entry) {
  const status = entry?.lifecycle?.status ?? {};
  const tested = entry?.lifecycle?.tested ?? null;
  return status.runtime === 'active'
    && status.conversion === 'ready'
    && (status.tested === 'verified' || tested?.result === 'pass');
}

function isActiveReadyModel(entry) {
  const status = entry?.lifecycle?.status ?? {};
  return status.runtime === 'active' && status.conversion === 'ready';
}

function selectCatalogModels(catalog, options) {
  const modelFilter = new Set(options.modelIds);
  const models = Array.isArray(catalog?.models) ? catalog.models : [];
  return models
    .filter((entry) => typeof entry?.modelId === 'string' && entry.modelId.length > 0)
    .filter((entry) => {
      if (modelFilter.size > 0 && !modelFilter.has(entry.modelId)) return false;
      if (options.allCatalog) return true;
      if (options.includeUntested) return isActiveReadyModel(entry);
      return isVerifiedSupportedModel(entry);
    })
    .sort((a, b) => a.modelId.localeCompare(b.modelId));
}

async function listLocalManifestIds(manifestRoot) {
  let entries;
  try {
    entries = await fs.readdir(manifestRoot, { withFileTypes: true });
  } catch (error) {
    if (error?.code === 'ENOENT') return [];
    throw error;
  }

  const ids = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    try {
      await fs.access(path.join(manifestRoot, entry.name, 'manifest.json'));
      ids.push(entry.name);
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
    }
  }
  return ids.sort();
}

function collectRegistryEntries(registry) {
  const entries = [];
  const byKey = new Map();
  const byWgsl = new Map();

  for (const [operation, opSchema] of Object.entries(registry?.operations ?? {})) {
    for (const [variant, variantDef] of Object.entries(opSchema?.variants ?? {})) {
      const wgsl = variantDef?.wgsl;
      if (typeof wgsl !== 'string' || wgsl.length === 0) continue;
      const entryPoint = variantDef?.entryPoint ?? 'main';
      const item = {
        operation,
        variant,
        wgsl,
        entryPoint,
        key: kernelKey(wgsl, entryPoint),
        reachabilityStatus: variantDef?.reachability?.status ?? null,
      };
      entries.push(item);
      pushMapList(byKey, item.key, item);
      pushMapList(byWgsl, wgsl, item);
    }
  }

  entries.sort(compareRegistryEntry);
  return { entries, byKey, byWgsl };
}

function compareRegistryEntry(a, b) {
  return `${a.operation}/${a.variant}`.localeCompare(`${b.operation}/${b.variant}`);
}

function pushMapList(map, key, value) {
  if (!map.has(key)) map.set(key, []);
  map.get(key).push(value);
}

function kernelKey(wgsl, entry) {
  return `${wgsl}#${entry}`;
}

async function loadSelectedManifests(selectedModels, manifestRoot) {
  const loaded = [];
  const missing = [];
  const invalid = [];

  for (const model of selectedModels) {
    const manifestPath = path.join(manifestRoot, model.modelId, 'manifest.json');
    let manifest;
    try {
      manifest = await readJson(manifestPath);
    } catch (error) {
      if (error?.code === 'ENOENT') {
        missing.push({ modelId: model.modelId, manifestPath: path.relative(repoRoot, manifestPath) });
        continue;
      }
      invalid.push({ modelId: model.modelId, manifestPath: path.relative(repoRoot, manifestPath), error: error.message });
      continue;
    }

    loaded.push({
      modelId: model.modelId,
      manifestPath: path.relative(repoRoot, manifestPath),
      kernels: collectManifestKernels(manifest),
      execution: manifest?.inference?.execution ?? null,
      capabilityContext: buildManifestCapabilityContext(model.modelId, manifest),
      lifecycle: {
        runtime: model.lifecycle?.status?.runtime ?? null,
        conversion: model.lifecycle?.status?.conversion ?? null,
        tested: model.lifecycle?.status?.tested ?? null,
        testedResult: model.lifecycle?.tested?.result ?? null,
      },
    });
  }

  return { loaded, missing, invalid };
}

async function loadOutsideScopeManifests(modelIds, manifestRoot) {
  const loaded = [];
  const invalid = [];

  for (const modelId of modelIds) {
    const manifestPath = path.join(manifestRoot, modelId, 'manifest.json');
    let manifest;
    try {
      manifest = await readJson(manifestPath);
    } catch (error) {
      invalid.push({ modelId, manifestPath: path.relative(repoRoot, manifestPath), error: error.message });
      continue;
    }

    loaded.push({
      modelId,
      manifestPath: path.relative(repoRoot, manifestPath),
      kernels: collectManifestKernels(manifest),
    });
  }

  return { loaded, invalid };
}

function collectManifestKernels(manifest) {
  return collectExecutionKernels(manifest?.inference?.execution);
}

function collectExecutionKernels(execution) {
  const kernels = execution?.kernels;
  if (!kernels || typeof kernels !== 'object' || Array.isArray(kernels)) return [];

  const refs = [];
  for (const [kernelRef, kernelDef] of Object.entries(kernels)) {
    const wgsl = kernelDef?.kernel;
    if (typeof wgsl !== 'string' || wgsl.length === 0) continue;
    const entry = kernelDef?.entry ?? kernelDef?.entryPoint ?? 'main';
    refs.push({
      kernelRef,
      wgsl,
      entry,
      key: kernelKey(wgsl, entry),
      digest: typeof kernelDef?.digest === 'string' ? kernelDef.digest : null,
    });
  }
  refs.sort((a, b) => a.kernelRef.localeCompare(b.kernelRef));
  return refs;
}

function buildManifestCapabilityContext(modelId, manifest) {
  const execution = manifest?.inference?.execution ?? null;
  const session = manifest?.inference?.session ?? {};
  const computeDefaults = session?.compute?.defaults ?? {};
  const activationDtype = computeDefaults.activationDtype ?? null;
  const mathDtype = computeDefaults.mathDtype ?? null;
  const accumDtype = computeDefaults.accumDtype ?? null;
  return {
    modelId,
    activationDtype,
    mathDtype,
    accumDtype,
    kvDtype: session?.kvcache?.kvDtype ?? activationDtype,
    headDim: Number.isFinite(manifest?.architecture?.headDim) ? manifest.architecture.headDim : null,
    layerTypes: Array.isArray(manifest?.inference?.layerPattern?.layerTypes)
      ? manifest.inference.layerPattern.layerTypes
      : null,
    retainQ4KMaterialization: session?.retainQ4KMaterialization === true,
    ...summarizeExecutionGraphContext(execution),
  };
}

function summarizeExecutionGraphContext(execution) {
  const summary = {
    hasDensePrefillProjectionKernel: false,
    hasQ4DecodeProjectionKernel: false,
    hasQ4PrefillProjectionKernel: false,
    hasAvailableQ4PrefillProjectionKernel: false,
    requiresF16ActivationNarrowing: false,
  };

  for (const [phase, steps] of [['decode', execution?.decode ?? []], ['prefill', execution?.prefill ?? []]]) {
    for (const step of flattenSteps(steps)) {
      if (!PROJECTION_OPS.has(step[0])) continue;
      const kernelEntry = execution?.kernels?.[step[1]];
      if (!kernelEntry) continue;
      if (phase === 'decode' && kernelEntry.kernel === 'fused_matmul_q4.wgsl') {
        summary.hasQ4DecodeProjectionKernel = true;
      }
      if (phase === 'prefill' && DENSE_Q4_PREFILL_FILES.has(kernelEntry.kernel)) {
        summary.hasDensePrefillProjectionKernel = true;
      }
      if (phase === 'prefill' && typeof kernelEntry.kernel === 'string' && kernelEntry.kernel.startsWith('fused_matmul_q4')) {
        summary.hasQ4PrefillProjectionKernel = true;
      }
    }
  }

  summary.hasAvailableQ4PrefillProjectionKernel = Object.values(execution?.kernels ?? {}).some(
    (entry) => entry?.kernel === 'fused_matmul_q4_batched_multicol_shared.wgsl'
      || entry?.kernel === 'fused_matmul_q4_batched.wgsl'
  );
  summary.requiresF16ActivationNarrowing = Object.values(execution?.kernels ?? {}).some(
    (entry) => F32_ACTIVATION_NARROWING_FILES.has(entry?.kernel)
  );
  return summary;
}

function flattenSteps(entries) {
  const steps = [];
  for (const entry of entries ?? []) {
    if (Array.isArray(entry)) {
      steps.push(entry);
      continue;
    }
    if (entry && typeof entry === 'object' && Array.isArray(entry.steps)) {
      steps.push(...flattenSteps(entry.steps));
    }
  }
  return steps;
}

function buildReport({ options, catalogPath, manifestRoot, registryPath, capabilityRulesPath, usageAllowlistPath, selectedModels, localManifestIds, loaded, missing, invalid, outsideScopeManifests, repoReferenceIndex, registryIndex, capabilityRules, usageAllowlist }) {
  const selectedModelIds = new Set(selectedModels.map((entry) => entry.modelId));
  const usedKeyToModels = new Map();
  const manifestKeyToModels = new Map();
  const capabilityKeyToModels = new Map();
  const usedWgslToModels = new Map();
  const usedKeyToSources = new Map();
  const outsideScopeKeyToModels = new Map();
  const outsideScopeWgslToModels = new Map();
  const manifestKernelsNotInRegistry = [];
  const capabilityKernelsNotInRegistry = [];
  const capabilityTransformApplications = [];

  for (const model of loaded) {
    for (const ref of model.kernels) {
      pushSetMap(usedKeyToModels, ref.key, model.modelId);
      pushSetMap(manifestKeyToModels, ref.key, model.modelId);
      pushSetMap(usedWgslToModels, ref.wgsl, model.modelId);
      pushSetMap(usedKeyToSources, ref.key, `manifest:${model.modelId}`);
      if (!registryIndex.byKey.has(ref.key)) {
        addKernelRefIssue(manifestKernelsNotInRegistry, {
          modelId: model.modelId,
          kernelRef: ref.kernelRef,
          wgsl: ref.wgsl,
          entry: ref.entry,
          key: ref.key,
          sources: ['manifest'],
        });
      }
    }
  }

  if (!options.manifestOnly) {
    const capabilityRefs = collectCapabilityTransformRefs(loaded, capabilityRules, options);
    for (const application of capabilityRefs.applications) {
      capabilityTransformApplications.push(application);
    }
    for (const ref of capabilityRefs.refs) {
      pushSetMap(usedKeyToModels, ref.key, ref.modelId);
      pushSetMap(capabilityKeyToModels, ref.key, ref.modelId);
      pushSetMap(usedWgslToModels, ref.wgsl, ref.modelId);
      pushSetMap(usedKeyToSources, ref.key, `capability:${ref.modelId}:${ref.transforms.join('+')}`);
      if (!registryIndex.byKey.has(ref.key)) {
        addKernelRefIssue(capabilityKernelsNotInRegistry, {
          modelId: ref.modelId,
          kernelRef: ref.kernelRef,
          wgsl: ref.wgsl,
          entry: ref.entry,
          key: ref.key,
          sources: [`capability:${ref.transforms.join('+')}`],
        });
      }
    }
  }

  for (const model of outsideScopeManifests.loaded) {
    for (const ref of model.kernels) {
      pushSetMap(outsideScopeKeyToModels, ref.key, model.modelId);
      pushSetMap(outsideScopeWgslToModels, ref.wgsl, model.modelId);
    }
  }

  const usedKeys = new Set(usedKeyToModels.keys());
  const manifestKeys = new Set(manifestKeyToModels.keys());
  const capabilityKeys = new Set(capabilityKeyToModels.keys());
  const usedWgslFiles = new Set(usedWgslToModels.keys());
  const manifestWgslFiles = new Set();
  for (const key of manifestKeys) {
    const [wgsl] = splitKernelKey(key);
    manifestWgslFiles.add(wgsl);
  }
  const registryWgslFiles = [...registryIndex.byWgsl.keys()].sort();
  const allowlistIndex = buildAllowlistIndex(registryIndex.entries, usageAllowlist, options);
  const classification = classifyRegistryEntries(registryIndex.entries, {
    usedKeys,
    outsideScopeKeyToModels,
    outsideScopeWgslToModels,
    repoReferenceIndex,
    allowlistIndex,
  });
  const allowlistedCategoryCounts = summarizeAllowlistedCategories(classification.allowlisted);
  const unusedRegistryVariants = registryIndex.entries
    .filter((entry) => !usedKeys.has(entry.key))
    .map(formatRegistryEntry);
  const unusedRegistryWgslFiles = registryWgslFiles
    .filter((wgsl) => !usedWgslFiles.has(wgsl))
    .map((wgsl) => ({
      wgsl,
      variants: (registryIndex.byWgsl.get(wgsl) ?? []).map(formatRegistryEntry).sort(compareFormattedVariant),
    }));
  const usedRegistryVariants = registryIndex.entries
    .filter((entry) => usedKeys.has(entry.key))
    .map(formatRegistryEntry);
  const manifestRegistryVariants = registryIndex.entries
    .filter((entry) => manifestKeys.has(entry.key))
    .map(formatRegistryEntry);
  const usedLiveKernels = [...usedKeyToModels.entries()]
    .map(([key, models]) => {
      const [wgsl, entry] = splitKernelKey(key);
      return {
        key,
        wgsl,
        entry,
        modelIds: [...models].sort(),
        sources: [...(usedKeyToSources.get(key) ?? [])].sort(),
        registryVariants: (registryIndex.byKey.get(key) ?? []).map(formatRegistryEntry).sort(compareFormattedVariant),
      };
    })
    .sort((a, b) => a.key.localeCompare(b.key));

  const localManifestIdsOutsideScope = localManifestIds
    .filter((modelId) => !selectedModelIds.has(modelId))
    .sort();

  return {
    scope: {
      catalogPath: path.relative(repoRoot, catalogPath),
      manifestRoot: path.relative(repoRoot, manifestRoot),
      registryPath: path.relative(repoRoot, registryPath),
      capabilityRulesPath: path.relative(repoRoot, capabilityRulesPath),
      usageAllowlistPath: options.noAllowlist ? null : path.relative(repoRoot, usageAllowlistPath),
      selection: selectionDescription(options),
      capabilityTransforms: options.manifestOnly ? 'excluded' : 'included',
      usageAllowlist: options.noAllowlist ? 'excluded' : 'included',
    },
    summary: {
      selectedCatalogModels: selectedModels.length,
      loadedManifests: loaded.length,
      missingManifests: missing.length,
      invalidManifests: invalid.length,
      localManifestsOutsideScope: localManifestIdsOutsideScope.length,
      outsideScopeManifestKernelKeys: outsideScopeKeyToModels.size,
      outsideScopeInvalidManifests: outsideScopeManifests.invalid.length,
      manifestKernelKeysUsed: manifestKeys.size,
      capabilityKernelKeysUsed: capabilityKeys.size,
      liveKernelKeysUsed: usedLiveKernels.length,
      capabilityTransformApplications: capabilityTransformApplications.length,
      registryVariantsUsedBySelectedScope: usedRegistryVariants.length,
      registryVariantsUnusedBySelectedScope: unusedRegistryVariants.length,
      registryVariantsOutsideSelectedModelScope: classification.outsideSelectedModelScope.length,
      registryVariantsRepoContractReferenced: classification.repoContractReferenced.length,
      registryVariantsJsDispatched: classification.jsDispatched.length,
      registryVariantsProtectedByAllowlist: classification.allowlisted.length,
      registryVariantsProtectedByReachability: classification.reachabilityProtected.length,
      deleteCandidateRegistryVariants: classification.deleteCandidates.length,
      unusedCandidateRegistryVariants: classification.deleteCandidates.length,
      registryWgslFilesUsedBySelectedScope: [...usedWgslFiles].filter((wgsl) => registryIndex.byWgsl.has(wgsl)).length,
      registryWgslFilesUnusedBySelectedScope: unusedRegistryWgslFiles.length,
      deleteCandidateRegistryWgslFiles: classification.deleteCandidateWgslFiles.length,
      unusedCandidateRegistryWgslFiles: classification.deleteCandidateWgslFiles.length,
      registryVariantsUsedBySelectedManifests: manifestRegistryVariants.length,
      registryWgslFilesUsedBySelectedManifests: [...manifestWgslFiles].filter((wgsl) => registryIndex.byWgsl.has(wgsl)).length,
      manifestKernelRefsNotInRegistry: manifestKernelsNotInRegistry.length,
      capabilityKernelRefsNotInRegistry: capabilityKernelsNotInRegistry.length,
      liveKernelRefsNotInRegistry: manifestKernelsNotInRegistry.length + capabilityKernelsNotInRegistry.length,
      staleAllowlistEntries: allowlistIndex.staleEntries.length,
      registryVariantsTotal: registryIndex.entries.length,
      registryWgslFilesTotal: registryWgslFiles.length,
    },
    lifecycleClassCounts: {
      'outside-selected-model-scope': classification.outsideSelectedModelScope.length,
      'repo-contract-referenced': classification.repoContractReferenced.length,
      'js-dispatched': classification.jsDispatched.length,
      'allowlisted': classification.allowlistedPrimary.length,
      'reachability-protected': classification.reachabilityProtected.length,
      'delete-candidate': classification.deleteCandidates.length,
    },
    referenceScan: {
      roots: repoReferenceIndex.roots,
      scannedFiles: repoReferenceIndex.scannedFiles,
    },
    allowlist: {
      enabled: !options.noAllowlist,
      path: options.noAllowlist ? null : path.relative(repoRoot, usageAllowlistPath),
      matchedEntries: allowlistIndex.matchedEntries,
      staleEntries: allowlistIndex.staleEntries,
      categoryCounts: allowlistedCategoryCounts,
    },
    selectedModels: selectedModels.map((entry) => ({
      modelId: entry.modelId,
      runtime: entry.lifecycle?.status?.runtime ?? null,
      conversion: entry.lifecycle?.status?.conversion ?? null,
      tested: entry.lifecycle?.status?.tested ?? null,
      testedResult: entry.lifecycle?.tested?.result ?? null,
    })),
    loadedManifests: loaded.map((model) => ({
      modelId: model.modelId,
      manifestPath: model.manifestPath,
      kernelRefs: model.kernels.length,
      lifecycle: model.lifecycle,
    })),
    missingManifests: missing,
    invalidManifests: invalid,
    localManifestIdsOutsideScope,
    outsideScopeManifests: outsideScopeManifests.loaded.map((model) => ({
      modelId: model.modelId,
      manifestPath: model.manifestPath,
      kernelRefs: model.kernels.length,
    })),
    outsideScopeInvalidManifests: outsideScopeManifests.invalid,
    usedLiveKernels,
    capabilityTransformApplications: capabilityTransformApplications.sort(compareCapabilityApplication),
    manifestKernelsNotInRegistry: sortKernelRefIssues(manifestKernelsNotInRegistry),
    capabilityKernelsNotInRegistry: sortKernelRefIssues(capabilityKernelsNotInRegistry),
    liveKernelsNotInRegistry: sortKernelRefIssues([
      ...manifestKernelsNotInRegistry,
      ...capabilityKernelsNotInRegistry,
    ]),
    outsideSelectedModelScopeRegistryVariants: classification.outsideSelectedModelScope,
    repoContractReferencedRegistryVariants: classification.repoContractReferenced,
    jsDispatchedRegistryVariants: classification.jsDispatched,
    allowlistedRegistryVariants: classification.allowlisted,
    reachabilityProtectedRegistryVariants: classification.reachabilityProtected,
    deleteCandidateRegistryVariants: classification.deleteCandidates,
    deleteCandidateRegistryWgslFiles: classification.deleteCandidateWgslFiles,
    unusedCandidateRegistryVariants: classification.deleteCandidates,
    unusedCandidateRegistryWgslFiles: classification.deleteCandidateWgslFiles,
    unusedRegistryVariants,
    unusedRegistryWgslFiles,
  };
}

function summarizeAllowlistedCategories(allowlistedVariants) {
  const counts = {};
  for (const variant of allowlistedVariants) {
    for (const entry of variant.allowlist ?? []) {
      const category = entry.category ?? 'uncategorized';
      counts[category] = (counts[category] ?? 0) + 1;
    }
  }
  return counts;
}

function buildAllowlistIndex(registryEntries, usageAllowlist, options) {
  if (options.noAllowlist) {
    return {
      matchesByEntryId: new Map(),
      matchedEntries: [],
      staleEntries: [],
      categoryCounts: {},
    };
  }

  const entries = Array.isArray(usageAllowlist?.entries) ? usageAllowlist.entries : [];
  const matchesByEntryId = new Map();
  const matchedEntries = [];
  const staleEntries = [];
  const categoryCounts = {};

  for (const allowEntry of entries) {
    validateAllowlistEntry(allowEntry);
    const matches = registryEntries.filter((registryEntry) => allowlistEntryMatchesRegistryEntry(allowEntry, registryEntry));
    const normalized = {
      id: allowEntry?.id ?? null,
      category: allowEntry?.category ?? 'uncategorized',
      reason: allowEntry?.reason ?? null,
      owner: allowEntry?.owner ?? null,
      matchedVariants: matches.length,
    };
    if (matches.length === 0) {
      staleEntries.push(normalized);
      continue;
    }
    matchedEntries.push(normalized);
    categoryCounts[normalized.category] = (categoryCounts[normalized.category] ?? 0) + matches.length;
    for (const registryEntry of matches) {
      const id = registryEntryId(registryEntry);
      if (!matchesByEntryId.has(id)) matchesByEntryId.set(id, []);
      matchesByEntryId.get(id).push(normalized);
    }
  }

  matchedEntries.sort((a, b) => String(a.id).localeCompare(String(b.id)));
  staleEntries.sort((a, b) => String(a.id).localeCompare(String(b.id)));
  return { matchesByEntryId, matchedEntries, staleEntries, categoryCounts };
}

function validateAllowlistEntry(allowEntry) {
  const id = allowEntry?.id ?? '<unknown>';
  if (typeof allowEntry?.id !== 'string' || allowEntry.id.length === 0) {
    throw new Error('Kernel usage allowlist entries must define a non-empty string id.');
  }
  const match = allowEntry?.match;
  if (!match || typeof match !== 'object' || Array.isArray(match)) {
    throw new Error(`Kernel usage allowlist entry "${id}" must define a match object.`);
  }
  const matchKeys = Object.keys(match);
  const unknownFields = matchKeys.filter((field) => !ALLOWLIST_MATCH_FIELDS.has(field));
  if (unknownFields.length > 0) {
    throw new Error(`Kernel usage allowlist entry "${id}" has unsupported match field(s): ${unknownFields.join(', ')}.`);
  }
  if (matchKeys.length === 0) {
    throw new Error(`Kernel usage allowlist entry "${id}" must define at least one match field.`);
  }
  for (const [field, values] of Object.entries(match)) {
    if (!Array.isArray(values) || values.length === 0 || values.some((value) => typeof value !== 'string' || value.length === 0)) {
      throw new Error(`Kernel usage allowlist entry "${id}" match.${field} must be a non-empty string array.`);
    }
  }
}

function allowlistEntryMatchesRegistryEntry(allowEntry, registryEntry) {
  const match = allowEntry.match;
  return matchesAny(match.keys, registryEntry.key)
    && matchesAny(match.operations, registryEntry.operation)
    && matchesAnyPrefix(match.operationPrefixes, registryEntry.operation)
    && matchesAnySuffix(match.operationSuffixes, registryEntry.operation)
    && matchesAny(match.variants, registryEntry.variant)
    && matchesAnyPrefix(match.variantPrefixes, registryEntry.variant)
    && matchesAnySuffix(match.variantSuffixes, registryEntry.variant)
    && matchesAny(match.wgsl, registryEntry.wgsl)
    && matchesAnyPrefix(match.wgslPrefixes, registryEntry.wgsl)
    && matchesAnySuffix(match.wgslSuffixes, registryEntry.wgsl)
    && matchesAny(match.reachabilityStatuses, registryEntry.reachabilityStatus);
}

function matchesAny(values, actual) {
  if (values === undefined) return true;
  return Array.isArray(values) && values.includes(actual);
}

function matchesAnyPrefix(values, actual) {
  if (values === undefined) return true;
  return Array.isArray(values) && values.some((value) => typeof actual === 'string' && actual.startsWith(value));
}

function matchesAnySuffix(values, actual) {
  if (values === undefined) return true;
  return Array.isArray(values) && values.some((value) => typeof actual === 'string' && actual.endsWith(value));
}

function classifyRegistryEntries(registryEntries, context) {
  const {
    usedKeys,
    outsideScopeKeyToModels,
    outsideScopeWgslToModels,
    repoReferenceIndex,
    allowlistIndex,
  } = context;
  const outsideSelectedModelScope = [];
  const repoContractReferenced = [];
  const jsDispatched = [];
  const allowlisted = [];
  const allowlistedPrimary = [];
  const reachabilityProtected = [];
  const deleteCandidates = [];
  const deleteCandidateIds = new Set();

  for (const registryEntry of registryEntries) {
    if (usedKeys.has(registryEntry.key)) continue;
    const id = registryEntryId(registryEntry);
    const allowlistMatches = allowlistIndex.matchesByEntryId.get(id) ?? [];
    const formatted = formatRegistryEntry(registryEntry);
    const withAllowlist = allowlistMatches.length > 0
      ? {
          ...formatted,
          allowlist: allowlistMatches.map(formatAllowlistMatch),
        }
      : formatted;
    if (allowlistMatches.length > 0) {
      allowlisted.push(withAllowlist);
    }

    const outsideScopeModelIds = [
      ...(outsideScopeKeyToModels.get(registryEntry.key) ?? []),
      ...(outsideScopeWgslToModels.get(registryEntry.wgsl) ?? []),
    ].sort();
    if (outsideScopeModelIds.length > 0) {
      outsideSelectedModelScope.push({
        ...withAllowlist,
        lifecycleClass: 'outside-selected-model-scope',
        outsideScopeModelIds: [...new Set(outsideScopeModelIds)],
      });
      continue;
    }

    const jsSources = [...(repoReferenceIndex.byEntryId.get(id) ?? [])].sort();
    if (jsSources.length > 0) {
      jsDispatched.push({
        ...withAllowlist,
        lifecycleClass: 'js-dispatched',
        sources: jsSources,
      });
      continue;
    }

    const repoContractSources = [...(repoReferenceIndex.byWgsl.get(registryEntry.wgsl) ?? [])].sort();
    if (repoContractSources.length > 0) {
      repoContractReferenced.push({
        ...withAllowlist,
        lifecycleClass: 'repo-contract-referenced',
        sources: repoContractSources,
      });
      continue;
    }

    if (allowlistMatches.length > 0) {
      allowlistedPrimary.push({
        ...withAllowlist,
        lifecycleClass: 'allowlisted',
      });
      continue;
    }

    if (REACHABILITY_PROTECTED_STATUSES.has(registryEntry.reachabilityStatus)) {
      reachabilityProtected.push({
        ...formatted,
        lifecycleClass: 'reachability-protected',
      });
      continue;
    }
    deleteCandidates.push({
      ...formatted,
      lifecycleClass: 'delete-candidate',
    });
    deleteCandidateIds.add(id);
  }

  const byWgsl = new Map();
  for (const registryEntry of registryEntries) {
    if (!byWgsl.has(registryEntry.wgsl)) byWgsl.set(registryEntry.wgsl, []);
    byWgsl.get(registryEntry.wgsl).push(registryEntry);
  }
  const deleteCandidateWgslFiles = [...byWgsl.entries()]
    .filter(([, variants]) => variants.every((entry) => deleteCandidateIds.has(registryEntryId(entry))))
    .map(([wgsl, variants]) => ({
      wgsl,
      variants: variants.map(formatRegistryEntry).sort(compareFormattedVariant),
    }))
    .sort((a, b) => a.wgsl.localeCompare(b.wgsl));

  return {
    outsideSelectedModelScope: outsideSelectedModelScope.sort(compareFormattedVariant),
    repoContractReferenced: repoContractReferenced.sort(compareFormattedVariant),
    jsDispatched: jsDispatched.sort(compareFormattedVariant),
    allowlisted: allowlisted.sort(compareFormattedVariant),
    allowlistedPrimary: allowlistedPrimary.sort(compareFormattedVariant),
    reachabilityProtected: reachabilityProtected.sort(compareFormattedVariant),
    deleteCandidates: deleteCandidates.sort(compareFormattedVariant),
    deleteCandidateWgslFiles,
  };
}

function formatAllowlistMatch(entry) {
  return {
    id: entry.id,
    category: entry.category,
    reason: entry.reason,
    owner: entry.owner,
  };
}

function registryEntryId(entry) {
  return `${entry.operation}/${entry.variant}`;
}

function addKernelRefIssue(issues, issue) {
  const existing = issues.find((entry) => {
    return entry.modelId === issue.modelId
      && entry.kernelRef === issue.kernelRef
      && entry.key === issue.key;
  });
  if (existing) {
    existing.sources = [...new Set([...(existing.sources ?? []), ...(issue.sources ?? [])])].sort();
    return;
  }
  issues.push({
    ...issue,
    sources: [...new Set(issue.sources ?? [])].sort(),
  });
}

function sortKernelRefIssues(issues) {
  return issues.sort((a, b) => `${a.modelId}:${a.kernelRef}:${a.key}`.localeCompare(`${b.modelId}:${b.kernelRef}:${b.key}`));
}

function collectCapabilityTransformRefs(loadedModels, capabilityRules, options) {
  const refs = [];
  const applications = [];
  const rules = Array.isArray(capabilityRules?.capabilityTransforms)
    ? capabilityRules.capabilityTransforms
    : [];

  for (const model of loadedModels) {
    if (!model.execution?.kernels || !model.capabilityContext) continue;
    const baseKeys = new Set(model.kernels.map((ref) => ref.key));
    for (const rule of rules) {
      if (!Array.isArray(rule?.transforms) || rule.transforms.length === 0) continue;
      const context = buildCapabilityRuleContext(rule, model.capabilityContext, options);
      if (!context) continue;
      const applied = applyCapabilityTransforms(model.execution, context, rule.transforms);
      if (!applied.changed) continue;
      const transformedRefs = collectExecutionKernels(applied.execution)
        .filter((ref) => !baseKeys.has(ref.key));
      if (transformedRefs.length === 0) continue;
      applications.push({
        modelId: model.modelId,
        transforms: rule.transforms,
        reason: rule.reason ?? null,
        addedKernelKeys: [...new Set(transformedRefs.map((ref) => ref.key))].sort(),
      });
      for (const ref of transformedRefs) {
        refs.push({
          ...ref,
          modelId: model.modelId,
          transforms: rule.transforms,
          reason: rule.reason ?? null,
        });
      }
    }
  }

  return { refs, applications };
}

function buildCapabilityRuleContext(rule, baseContext, options) {
  const match = rule?.match ?? {};
  if (!isRuleCompatibleWithModel(match, baseContext, options)) {
    return null;
  }

  const context = {
    capabilities: {
      hasSubgroups: true,
      hasF16: true,
      hasSubgroupsF16: true,
      maxWorkgroupStorageSize: 32768,
    },
    platform: {
      id: 'synthetic',
      vendor: 'unknown',
      architecture: 'unknown',
    },
    graphContext: {
      ...baseContext,
    },
  };

  for (const [key, condition] of Object.entries(match)) {
    applyRuleConditionToContext(context, key, condition);
  }
  context.capabilities.hasSubgroupsF16 = context.capabilities.hasSubgroups === true && context.capabilities.hasF16 === true;

  return {
    capabilities: context.capabilities,
    platform: context.platform,
    transformContext: {
      capabilities: context.capabilities,
      platform: context.platform,
      activationDtype: context.graphContext.activationDtype,
      mathDtype: context.graphContext.mathDtype,
      accumDtype: context.graphContext.accumDtype,
      kvDtype: context.graphContext.kvDtype,
      headDim: context.graphContext.headDim,
      modelId: context.graphContext.modelId,
      layerTypes: context.graphContext.layerTypes,
      retainQ4KMaterialization: context.graphContext.retainQ4KMaterialization,
    },
  };
}

function isRuleCompatibleWithModel(match, baseContext, options) {
  for (const [key, condition] of Object.entries(match ?? {})) {
    if (key === 'modelId') {
      if (!matchesStaticCondition(baseContext.modelId, condition, true)) {
        return false;
      }
      continue;
    }
    if (key === 'platformVendor' || key === 'platformId' || key === 'platformArchitecture') {
      continue;
    }
    if (RUNTIME_CONTEXT_KEYS.has(key) && !options.includeRuntimeTransformVariants) {
      if (!matchesStaticCondition(baseContext[key], condition, false)) {
        return false;
      }
      continue;
    }
    if (GRAPH_CONTEXT_KEYS.has(key) && !matchesStaticCondition(baseContext[key], condition, false)) {
      return false;
    }
  }
  return true;
}

function matchesStaticCondition(value, condition, allowMissing) {
  if (value == null) return allowMissing;
  if (condition == null || typeof condition !== 'object' || Array.isArray(condition)) {
    return value === condition;
  }
  if (condition.startsWith !== undefined) {
    return typeof value === 'string' && value.startsWith(condition.startsWith);
  }
  if (condition.contains !== undefined) {
    return typeof value === 'string' && value.includes(condition.contains);
  }
  if (condition.gte !== undefined) {
    return Number.isFinite(value) && value >= condition.gte;
  }
  if (condition.lte !== undefined) {
    return Number.isFinite(value) && value <= condition.lte;
  }
  return true;
}

function applyRuleConditionToContext(context, key, condition) {
  if (CAPABILITY_KEYS.has(key)) {
    context.capabilities[key] = materializeRuleValue(condition, context.capabilities[key]);
    return;
  }
  const platformKey = PLATFORM_KEY_MAP.get(key);
  if (platformKey) {
    context.platform[platformKey] = materializeRuleValue(condition, context.platform[platformKey]);
    return;
  }
  if (RUNTIME_CONTEXT_KEYS.has(key) || GRAPH_CONTEXT_KEYS.has(key)) {
    context.graphContext[key] = materializeRuleValue(condition, context.graphContext[key]);
  }
}

function materializeRuleValue(condition, fallback) {
  if (condition == null || typeof condition !== 'object' || Array.isArray(condition)) {
    return condition;
  }
  if (condition.gte !== undefined) return condition.gte;
  if (condition.lte !== undefined) return condition.lte;
  return fallback;
}

function applyCapabilityTransforms(execution, context, transformNames) {
  let current = execution;
  let changed = false;
  for (const name of transformNames) {
    const transform = TRANSFORMS[name];
    if (!transform) {
      throw new Error(`Unknown capability transform "${name}".`);
    }
    const result = transform(current, context.transformContext);
    if (result !== null && result !== undefined) {
      current = result;
      if (!SESSION_ONLY_TRANSFORMS.has(name)) {
        changed = true;
      }
    }
  }
  return { execution: current, changed };
}

function compareCapabilityApplication(a, b) {
  return `${a.modelId}:${a.transforms.join('+')}`.localeCompare(`${b.modelId}:${b.transforms.join('+')}`);
}

function splitKernelKey(key) {
  const index = key.lastIndexOf('#');
  if (index === -1) return [key, ''];
  return [key.slice(0, index), key.slice(index + 1)];
}

function pushSetMap(map, key, value) {
  if (!map.has(key)) map.set(key, new Set());
  map.get(key).add(value);
}

function formatRegistryEntry(entry) {
  return {
    operation: entry.operation,
    variant: entry.variant,
    wgsl: entry.wgsl,
    entryPoint: entry.entryPoint,
    key: entry.key,
    reachabilityStatus: entry.reachabilityStatus,
  };
}

function compareFormattedVariant(a, b) {
  return `${a.operation}/${a.variant}`.localeCompare(`${b.operation}/${b.variant}`);
}

function selectionDescription(options) {
  if (options.allCatalog) return 'all catalog entries';
  if (options.includeUntested) return 'catalog entries with lifecycle.status runtime=active and conversion=ready';
  return 'catalog entries with runtime=active, conversion=ready, and tested=verified/pass';
}

function printHumanReport(report, limit) {
  const s = report.summary;
  console.log(`[kernels:supported-manifests:report] ${report.scope.selection}`);
  console.log(`  selected catalog models: ${s.selectedCatalogModels}`);
  console.log(`  loaded local manifests:  ${s.loadedManifests}`);
  console.log(`  missing manifests:       ${s.missingManifests}`);
  console.log(`  invalid manifests:       ${s.invalidManifests}`);
  console.log(`  manifest kernel keys:    ${s.manifestKernelKeysUsed}`);
  console.log(`  capability kernel keys:  ${s.capabilityKernelKeysUsed}`);
  console.log(`  live kernel keys:        ${s.liveKernelKeysUsed}`);
  console.log(`  capability transforms with added kernels: ${s.capabilityTransformApplications}`);
  console.log(`  registry variants used:  ${s.registryVariantsUsedBySelectedScope}/${s.registryVariantsTotal}`);
  console.log(`  registry variants unused by selected scope: ${s.registryVariantsUnusedBySelectedScope}`);
  console.log(`  outside selected model scope: ${s.registryVariantsOutsideSelectedModelScope}`);
  console.log(`  repo-contract referenced: ${s.registryVariantsRepoContractReferenced}`);
  console.log(`  JS-dispatched: ${s.registryVariantsJsDispatched}`);
  console.log(`  protected by usage allowlist: ${s.registryVariantsProtectedByAllowlist}`);
  console.log(`  protected by registry reachability: ${s.registryVariantsProtectedByReachability}`);
  console.log(`  delete candidate variants: ${s.deleteCandidateRegistryVariants}`);
  console.log(`  registry WGSL files used: ${s.registryWgslFilesUsedBySelectedScope}/${s.registryWgslFilesTotal}`);
  console.log(`  registry WGSL files unused by selected scope: ${s.registryWgslFilesUnusedBySelectedScope}`);
  console.log(`  delete candidate WGSL files: ${s.deleteCandidateRegistryWgslFiles}`);
  console.log(`  manifest kernel refs absent from registry: ${s.manifestKernelRefsNotInRegistry}`);
  console.log(`  capability kernel refs absent from registry: ${s.capabilityKernelRefsNotInRegistry}`);
  console.log(`  stale allowlist entries: ${s.staleAllowlistEntries}`);
  console.log(`  local manifests outside scope: ${s.localManifestsOutsideScope}`);
  console.log(`  reference scan files: repo-contract=${report.referenceScan.scannedFiles.repoContract}, JS-dispatch=${report.referenceScan.scannedFiles.jsDispatch}`);

  printList('Selected models', report.selectedModels.map((entry) => entry.modelId), limit);
  printList('Local manifests outside scope', report.localManifestIdsOutsideScope, limit);
  printList('Lifecycle classes', Object.entries(report.lifecycleClassCounts).map(([name, count]) => `${name}: ${count}`), limit);
  printList('Capability transform applications', report.capabilityTransformApplications.map((entry) => `${entry.modelId}: ${entry.transforms.join(' + ')} (${entry.addedKernelKeys.length} key${entry.addedKernelKeys.length === 1 ? '' : 's'})`), limit);
  printList('Manifest kernels absent from registry', report.manifestKernelsNotInRegistry.map((entry) => `${entry.modelId}:${entry.kernelRef} -> ${entry.key}`), limit);
  printList('Capability kernels absent from registry', report.capabilityKernelsNotInRegistry.map((entry) => `${entry.modelId}:${entry.kernelRef} -> ${entry.key} [${entry.sources.join(', ')}]`), limit);
  printList('Stale allowlist entries', report.allowlist.staleEntries.map((entry) => `${entry.id} (${entry.category})`), limit);
  printList('Allowlist categories', Object.entries(report.allowlist.categoryCounts).sort(([a], [b]) => a.localeCompare(b)).map(([category, count]) => `${category}: ${count}`), limit);
  printList('Outside-scope manifest variants', report.outsideSelectedModelScopeRegistryVariants.map((entry) => `${entry.operation}/${entry.variant} -> ${entry.key} [${entry.outsideScopeModelIds.join(', ')}]`), limit);
  printList('Repo-contract referenced variants', report.repoContractReferencedRegistryVariants.map((entry) => `${entry.operation}/${entry.variant} -> ${entry.key} [${entry.sources.slice(0, 3).join(', ')}${entry.sources.length > 3 ? ', ...' : ''}]`), limit);
  printList('JS-dispatched variants', report.jsDispatchedRegistryVariants.map((entry) => `${entry.operation}/${entry.variant} -> ${entry.key} [${entry.sources.join(', ')}]`), limit);
  printList('Delete candidate WGSL files', report.deleteCandidateRegistryWgslFiles.map((entry) => `${entry.wgsl} (${entry.variants.length} variant${entry.variants.length === 1 ? '' : 's'})`), limit);
  printList('Delete candidate variants', report.deleteCandidateRegistryVariants.map((entry) => `${entry.operation}/${entry.variant} -> ${entry.key}`), limit);
  printList('Allowlisted non-live variants', report.allowlistedRegistryVariants.map((entry) => `${entry.operation}/${entry.variant} -> ${entry.key} [${entry.allowlist.map((item) => item.id).join(', ')}]`), limit);
  printList('Reachability-protected non-live variants', report.reachabilityProtectedRegistryVariants.map((entry) => `${entry.operation}/${entry.variant} -> ${entry.key} [${entry.reachabilityStatus}]`), limit);

  if (s.localManifestsOutsideScope > 0) {
    console.log(`  note: ${scopeHint(report.scope.selection)}`);
  }
  console.log('  note: delete candidates are non-live and have no selected-manifest, outside-scope manifest, repo-contract, JS-dispatch, allowlist, or protected-reachability evidence. Capability-transform outputs are included unless --manifest-only is set. Use --json for full lists.');
}

function printList(label, values, limit) {
  if (limit === 0 || values.length === 0) return;
  console.log(`${label}:`);
  for (const value of values.slice(0, limit)) {
    console.log(`  - ${value}`);
  }
  if (values.length > limit) {
    console.log(`  ... ${values.length - limit} more`);
  }
}

function scopeHint(selection) {
  if (selection === 'all catalog entries') {
    return 'local manifests outside scope are not present in models/catalog.json.';
  }
  if (selection.includes('runtime=active and conversion=ready')) {
    return 'use --all-catalog to include experimental or otherwise non-active catalog entries.';
  }
  return 'use --include-untested to include active/ready unverified local manifests, or --all-catalog for every catalog entry.';
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  const catalogPath = resolveRepoPath(options.catalogPath);
  const manifestRoot = resolveRepoPath(options.manifestRoot);
  const registryPath = resolveRepoPath(options.registryPath);
  const capabilityRulesPath = resolveRepoPath(options.capabilityRulesPath);
  const usageAllowlistPath = resolveRepoPath(options.usageAllowlistPath);

  const [catalog, registry, capabilityRules, usageAllowlist, localManifestIds] = await Promise.all([
    readJson(catalogPath),
    readJson(registryPath),
    readJson(capabilityRulesPath),
    options.noAllowlist ? Promise.resolve(null) : readJson(usageAllowlistPath),
    listLocalManifestIds(manifestRoot),
  ]);

  const selectedModels = selectCatalogModels(catalog, options);
  const selectedModelIds = new Set(selectedModels.map((entry) => entry.modelId));
  const localManifestIdsOutsideScope = localManifestIds
    .filter((modelId) => !selectedModelIds.has(modelId))
    .sort();
  const registryIndex = collectRegistryEntries(registry);
  const [manifests, outsideScopeManifests, repoReferenceIndex] = await Promise.all([
    loadSelectedManifests(selectedModels, manifestRoot),
    loadOutsideScopeManifests(localManifestIdsOutsideScope, manifestRoot),
    collectRepoReferenceIndex(registryIndex.entries),
  ]);
  const report = buildReport({
    options,
    catalogPath,
    manifestRoot,
    registryPath,
    capabilityRulesPath,
    usageAllowlistPath,
    selectedModels,
    localManifestIds,
    outsideScopeManifests,
    repoReferenceIndex,
    ...manifests,
    registryIndex,
    capabilityRules,
    usageAllowlist,
  });

  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    printHumanReport(report, options.limit);
  }

  const checkFailed = options.check && (
    report.summary.missingManifests > 0
    || report.summary.invalidManifests > 0
    || report.summary.liveKernelRefsNotInRegistry > 0
    || report.summary.staleAllowlistEntries > 0
  );
  const unusedCandidateGateFailed = (options.failOnUnusedCandidates || options.failOnDeleteCandidates)
    && report.summary.deleteCandidateRegistryVariants > 0;
  if (checkFailed || unusedCandidateGateFailed) {
    process.exitCode = 1;
  }
}

try {
  await main();
} catch (error) {
  console.error(`[kernels:supported-manifests:report] ${error.message}`);
  process.exit(1);
}
