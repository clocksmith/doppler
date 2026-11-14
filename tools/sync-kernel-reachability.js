#!/usr/bin/env node

/**
 * Computes and writes `reachability` metadata into each variant in
 * src/config/kernels/registry.json.
 *
 * For every variant the tool resolves:
 *   - Which conversion configs inline it (via execution.kernels)
 *   - Which runtime profiles pin it (via execution patches)
 *   - Which rule chains can select it (via src/rules/kernels/*.rules.json)
 *   - Which Rig recipes pin it (src/config/forge, counted with inline configs)
 *   - Whether shipped JavaScript dispatches it directly (tools/lib/kernel-code-references.js)
 *   - Whether only repository tests, tools or benchmarks name it
 *   - Whether the WGSL file exists on disk
 *   - A status classification: pinned, model-selectable, selectable,
 *     code-dispatched, repository-evidence, unused, missing-wgsl
 *
 * Usage:
 *   npm run kernels:registry:sync
 *   npm run kernels:registry:check
 *   node tools/sync-kernel-reachability.js --check --explain <operation/variant>
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  collectForgeRecipeReferences,
  collectKernelCodeReferences,
  collectRepositoryEvidenceReferences,
} from './lib/kernel-code-references.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const registryPath = path.join(repoRoot, 'src/config/kernels/registry.json');
const kernelsDir = path.join(repoRoot, 'src/gpu/kernels');
const rulesDir = path.join(repoRoot, 'src/rules/kernels');
const rulesRoot = path.join(repoRoot, 'src/rules');
const conversionRoot = path.join(repoRoot, 'src/config/conversion');
const runtimeRoot = path.join(repoRoot, 'src/config/runtime');

const checkMode = process.argv.includes('--check');
const explainIndex = process.argv.indexOf('--explain');
const explainKey = explainIndex >= 0 ? process.argv[explainIndex + 1] : null;
const INLINE_PRIMITIVE_ARRAY_MAX_CHARS = 80;
const ID_PATTERN = /^[a-z0-9]+(?:_[a-z0-9]+)*$/u;
const WGSL_SEGMENT_PATTERN = /^[a-z0-9]+(?:_[a-z0-9]+)*(?:\.wgsl)?$/u;

// ---------------------------------------------------------------------------
// Pass 1: Collect inline references from conversion configs
// ---------------------------------------------------------------------------

function addRef(refs, key, sourceId) {
  if (!refs.has(key)) refs.set(key, new Set());
  refs.get(key).add(sourceId);
}

function stringifyRegistry(registry) {
  const replacements = [];
  let replacementId = 0;
  let content = JSON.stringify(registry, (key, value) => {
    if (
      Array.isArray(value)
      && value.every((entry) => (
        entry === null
        || typeof entry === 'string'
        || typeof entry === 'number'
        || typeof entry === 'boolean'
      ))
    ) {
      const inline = JSON.stringify(value);
      if (inline.length <= INLINE_PRIMITIVE_ARRAY_MAX_CHARS) {
        const token = `__DOPPLER_INLINE_ARRAY_${replacementId}__`;
        replacementId += 1;
        replacements.push({ token, inline });
        return token;
      }
    }
    return value;
  }, 1);
  for (const { token, inline } of replacements) {
    content = content.replace(JSON.stringify(token), inline);
  }
  return content + '\n';
}

function collectKernelRefs(value, refs, sourceId) {
  if (Array.isArray(value)) {
    for (const entry of value) collectKernelRefs(entry, refs, sourceId);
    return;
  }
  if (!value || typeof value !== 'object') return;
  if (typeof value.kernel === 'string' && value.kernel.endsWith('.wgsl')) {
    const key = `${value.kernel}#${value.entry ?? 'main'}`;
    addRef(refs, key, sourceId);
  }
  for (const entry of Object.values(value)) {
    collectKernelRefs(entry, refs, sourceId);
  }
}

function collectConfiguredVariantRefs(value, refs, sourceId) {
  if (Array.isArray(value)) {
    for (const entry of value) collectConfiguredVariantRefs(entry, refs, sourceId);
    return;
  }
  if (!value || typeof value !== 'object') return;
  const fusedFfnQ4K = value.fusedFfnQ4K;
  if (fusedFfnQ4K && typeof fusedFfnQ4K === 'object') {
    for (const phaseConfig of Object.values(fusedFfnQ4K)) {
      if (typeof phaseConfig?.variant === 'string' && phaseConfig.variant.length > 0) {
        addRef(refs, `fused_ffn:${phaseConfig.variant}`, sourceId);
      }
    }
  }
  for (const entry of Object.values(value)) {
    collectConfiguredVariantRefs(entry, refs, sourceId);
  }
}

async function collectInlineRefs(configuredVariantRefs) {
  const refs = new Map(); // key = "wgsl#entry" -> Set<configId>
  for (const filePath of await walkJsonFiles(conversionRoot)) {
    const config = JSON.parse(await fs.readFile(filePath, 'utf8'));
    const configId = path.relative(conversionRoot, filePath).replace(/\.json$/, '');
    collectKernelRefs(config?.execution?.kernels, refs, configId);
  }
  for (const filePath of await walkJsonFiles(runtimeRoot)) {
    const config = JSON.parse(await fs.readFile(filePath, 'utf8'));
    const configId = `runtime/${path.relative(runtimeRoot, filePath).replace(/\.json$/, '')}`;
    collectKernelRefs(config, refs, configId);
    collectConfiguredVariantRefs(config, configuredVariantRefs, configId);
  }
  return refs;
}

async function walkJsonFiles(dir) {
  const results = [];
  const items = await fs.readdir(dir, { withFileTypes: true });
  for (const item of items) {
    const full = path.join(dir, item.name);
    if (item.isDirectory()) {
      results.push(...await walkJsonFiles(full));
    } else if (item.name.endsWith('.json')) {
      results.push(full);
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Pass 2: Collect rule-based references
// ---------------------------------------------------------------------------

async function collectRuleRefs(registry) {
  const registryOpNames = new Set(Object.keys(registry.operations ?? {}));
  const variantsByOp = new Map(Object.entries(registry.operations ?? {})
    .map(([op, schema]) => [op, new Set(Object.keys(schema.variants ?? {}))]));
  const refs = new Map(); // key = "operation:variantName" -> Set<"file#chain">
  const addScoped = (op, varName, source) => {
    const key = `${op}:${varName}`;
    if (!refs.has(key)) refs.set(key, new Set());
    refs.get(key).add(source);
  };
  const ruleFiles = (await fs.readdir(rulesDir))
    .filter((f) => f.endsWith('.rules.json'));

  for (const file of ruleFiles) {
    // Infer target operation(s) from filename: kebab-to-snake
    const ruleBase = file.replace('.rules.json', '').replace(/-/g, '_');
    const fileOps = registryOpNames.has(ruleBase)
      ? [ruleBase]
      : [...registryOpNames].filter((op) => op.startsWith(ruleBase + '_'));

    const rules = JSON.parse(await fs.readFile(path.join(rulesDir, file), 'utf8'));
    // `{ template: "{base}{suffix}" }` values expand placeholders from the
    // chain of the same name, or from every terminal value in the file when
    // the placeholder is a caller context (base comes from the phase chain).
    const chainValues = new Map();
    for (const [chainName, chain] of Object.entries(rules)) {
      if (!Array.isArray(chain)) continue;
      const values = new Map();
      for (const rule of chain) extractVariantNames(rule?.value, values, chainName);
      chainValues.set(chainName, [...values.keys()]);
    }
    const fileValues = [...new Set([...chainValues.values()].flat())];
    for (const [chainName, chain] of Object.entries(rules)) {
      if (!Array.isArray(chain)) continue;
      const source = `${file}#${chainName}`;
      const rawRefs = new Map(); // variantName -> Set<source>
      for (const rule of chain) {
        // Compound dispatch plans explicitly bind operation and variant.
        const operation = rule?.value?.operation;
        const variant = rule?.value?.variant;
        if (typeof operation === 'string' && typeof variant === 'string') {
          if (!variantsByOp.get(operation)?.has(variant)) {
            throw new Error(`Unknown kernel dispatch plan ${operation}:${variant} in ${source}`);
          }
          addScoped(operation, variant, source);
          continue;
        }
        extractVariantNames(rule?.value, rawRefs, source);
        const template = rule?.value?.template;
        if (typeof template === 'string') {
          for (const name of expandRuleTemplate(template, chainValues, fileValues)) {
            if (!rawRefs.has(name)) rawRefs.set(name, new Set());
            rawRefs.get(name).add(source);
          }
        }
      }
      const values = [...rawRefs.keys()].filter((name) => [...variantsByOp.values()].some((set) => set.has(name)));
      // A chain whose variant names all belong to an operation other than the
      // filename's (moe.rules.json#scatterAddVariant selects scatter_add) is
      // scoped to every operation that registers all of those names.
      const coversChain = (op) => values.length > 0 && values.every((name) => variantsByOp.get(op).has(name));
      const targetOps = fileOps.some(coversChain)
        ? fileOps
        : [...new Set([...fileOps, ...[...registryOpNames].filter(coversChain)])];
      for (const varName of rawRefs.keys()) {
        for (const op of targetOps) addScoped(op, varName, source);
      }
    }
  }
  // Rule assets outside src/rules/kernels may select a shader file directly
  // (inference/moe.rules.json#biasAddCode returns { shaderFile }).
  const wgslVariants = new Map();
  for (const [op, schema] of Object.entries(registry.operations ?? {})) {
    for (const [varName, varDef] of Object.entries(schema.variants ?? {})) {
      if (!wgslVariants.has(varDef.wgsl)) wgslVariants.set(varDef.wgsl, []);
      wgslVariants.get(varDef.wgsl).push({ op, varName, entry: varDef.entryPoint ?? 'main' });
    }
  }
  const allRuleFiles = (await walkJsonFiles(rulesRoot))
    .filter((f) => f.endsWith('.rules.json') && !f.includes(`${path.sep}generated${path.sep}`));
  for (const filePath of allRuleFiles) {
    const relative = path.relative(rulesRoot, filePath).split(path.sep).join('/');
    const rules = JSON.parse(await fs.readFile(filePath, 'utf8'));
    for (const [chainName, chain] of Object.entries(rules)) {
      const visit = (value) => {
        if (Array.isArray(value)) return value.forEach(visit);
        if (typeof value === 'string' && wgslVariants.has(value)) {
          const owners = wgslVariants.get(value);
          for (const owner of owners) {
            if (owners.length === 1 || owner.entry === 'main') {
              addScoped(owner.op, owner.varName, `${relative}#${chainName}`);
            }
          }
          return;
        }
        if (value && typeof value === 'object') Object.values(value).forEach(visit);
      };
      visit(chain);
    }
  }
  return refs;
}

function expandRuleTemplate(template, chainValues, fileValues) {
  let names = [''];
  for (const part of template.split(/(\{[a-zA-Z0-9_]+\})/u)) {
    if (!part) continue;
    const placeholder = /^\{([a-zA-Z0-9_]+)\}$/u.exec(part)?.[1];
    const options = placeholder ? ['', ...(chainValues.get(placeholder) ?? fileValues)] : [part];
    names = names.flatMap((prefix) => options.map((option) => prefix + option));
  }
  return [...new Set(names)];
}

function extractVariantNames(value, refs, source) {
  if (typeof value === 'string') {
    if (!refs.has(value)) refs.set(value, new Set());
    refs.get(value).add(source);
    return;
  }
  if (value && typeof value === 'object') {
    // Compound selections like { variant: { context: "..." }, useGemv: true }
    // or { context: "..." } are intermediate references, not terminal variant names.
    // Walk all string leaf values that look like variant names (skip context/template refs).
    for (const [k, v] of Object.entries(value)) {
      if (k === 'context' || k === 'template') continue;
      if (typeof v === 'string' && !['true', 'false'].includes(v)) {
        extractVariantNames(v, refs, source);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Pass 3: Check WGSL file existence
// ---------------------------------------------------------------------------

async function collectWgslOnDisk() {
  const files = new Set();
  async function walk(dir, prefix = '') {
    const items = await fs.readdir(dir, { withFileTypes: true });
    for (const item of items) {
      const relativePath = prefix ? `${prefix}/${item.name}` : item.name;
      const fullPath = path.join(dir, item.name);
      if (item.isDirectory()) {
        await walk(fullPath, relativePath);
        continue;
      }
      if (item.isFile() && item.name.endsWith('.wgsl')) {
        files.add(relativePath);
      }
    }
  }
  await walk(kernelsDir);
  return files;
}

function isSnakeCaseId(value) {
  return typeof value === 'string' && ID_PATTERN.test(value);
}

function isSnakeCaseWgslPath(value) {
  if (typeof value !== 'string' || !value.endsWith('.wgsl')) return false;
  return value.split('/').every((segment) => WGSL_SEGMENT_PATTERN.test(segment));
}

function collectInventoryIssues(registry, wgslOnDisk, unregistered) {
  const issues = [];
  const operations = registry.operations;
  if (!operations || typeof operations !== 'object' || Array.isArray(operations)) {
    issues.push('registry.operations must be an object.');
    return issues;
  }

  for (const [opName, opSchema] of Object.entries(operations)) {
    if (!isSnakeCaseId(opName)) {
      issues.push(`operation id "${opName}" is not snake_case.`);
    }

    const variants = opSchema?.variants;
    if (!variants || typeof variants !== 'object' || Array.isArray(variants)) {
      issues.push(`operation "${opName}" must define variants.`);
      continue;
    }

    for (const [varName, varDef] of Object.entries(variants)) {
      if (!isSnakeCaseId(varName)) {
        issues.push(`variant id "${opName}/${varName}" is not snake_case.`);
      }
      if (typeof varDef?.wgsl !== 'string' || !varDef.wgsl.trim()) {
        issues.push(`variant "${opName}/${varName}" is missing wgsl.`);
        continue;
      }
      if (!isSnakeCaseWgslPath(varDef.wgsl)) {
        issues.push(`variant "${opName}/${varName}" references non-snake-case WGSL path: ${varDef.wgsl}`);
      }
      if (!wgslOnDisk.has(varDef.wgsl)) {
        issues.push(`variant "${opName}/${varName}" references missing WGSL file: ${varDef.wgsl}`);
      }
      if (typeof varDef?.entryPoint !== 'string' || !varDef.entryPoint.trim()) {
        issues.push(`variant "${opName}/${varName}" is missing entryPoint.`);
      }
    }
  }

  for (const wgsl of unregistered) {
    issues.push(`WGSL file is not registered: ${wgsl}`);
  }

  return issues;
}

// ---------------------------------------------------------------------------
// Build reverse index: (wgsl, entry) -> (operation, variant)
// ---------------------------------------------------------------------------

function buildReverseIndex(registry) {
  const index = new Map(); // key = "wgsl#entry" -> [{ operation, variant }]
  for (const [opName, opSchema] of Object.entries(registry.operations ?? {})) {
    for (const [varName, varDef] of Object.entries(opSchema.variants ?? {})) {
      const wgsl = varDef?.wgsl;
      const entry = varDef?.entryPoint ?? 'main';
      if (typeof wgsl !== 'string') continue;
      const key = `${wgsl}#${entry}`;
      if (!index.has(key)) index.set(key, []);
      index.get(key).push({ operation: opName, variant: varName });
    }
  }
  return index;
}

// ---------------------------------------------------------------------------
// Pass 4: Determine which operations are exercised by actual models
// ---------------------------------------------------------------------------

function collectModelExercisedOps(inlineRefs, reverseIndex) {
  const ops = new Set();
  for (const wgslEntryKey of inlineRefs.keys()) {
    const entries = reverseIndex.get(wgslEntryKey);
    if (entries) {
      for (const { operation } of entries) ops.add(operation);
    }
  }
  return ops;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const registry = JSON.parse(await fs.readFile(registryPath, 'utf8'));
const registryOpNames = new Set(Object.keys(registry.operations ?? {}));
const configuredVariantRefs = new Map();
const inlineRefs = await collectInlineRefs(configuredVariantRefs);
const ruleRefs = await collectRuleRefs(registry);
const wgslOnDisk = await collectWgslOnDisk();
const reverseIndex = buildReverseIndex(registry);
const forgeRefs = await collectForgeRecipeReferences({ repoRoot, registry, collectKernelRefs });
for (const [key, sources] of forgeRefs.inlineRefs) {
  for (const source of sources) addRef(inlineRefs, key, source);
}
for (const [key, sources] of forgeRefs.pairRefs) {
  for (const source of sources) addRef(configuredVariantRefs, key, source);
}
const modelExercisedOps = collectModelExercisedOps(inlineRefs, reverseIndex);
const codeRefs = await collectKernelCodeReferences({ repoRoot, registry });
const evidenceRefs = await collectRepositoryEvidenceReferences({ repoRoot, registry });

let totalVariants = 0;
const statusCounts = {
  'pinned': 0,
  'model-selectable': 0,
  'selectable': 0,
  'code-dispatched': 0,
  'repository-evidence': 0,
  'unused': 0,
  'missing-wgsl': 0,
};

for (const [opName, opSchema] of Object.entries(registry.operations ?? {})) {
  for (const [varName, varDef] of Object.entries(opSchema.variants ?? {})) {
    totalVariants++;
    const wgsl = varDef?.wgsl;
    const entry = varDef?.entryPoint ?? 'main';
    const fileExists = typeof wgsl === 'string' && wgslOnDisk.has(wgsl);

    // Inline config references (match by wgsl+entry)
    const key = typeof wgsl === 'string' ? `${wgsl}#${entry}` : null;
    const inlineConfigSet = key ? (inlineRefs.get(key) ?? new Set()) : new Set();

    // Rule references (scoped by operation:variant)
    const ruleRefSet = ruleRefs.get(`${opName}:${varName}`) ?? new Set();

    const configuredVariantSet = configuredVariantRefs.get(`${opName}:${varName}`) ?? new Set();
    const inlineConfigs = [...new Set([...inlineConfigSet, ...configuredVariantSet])].sort();
    const ruleChains = [...ruleRefSet].sort();
    const dispatchSources = [...(codeRefs.get(`${opName}:${varName}`) ?? [])].sort();
    const evidenceSources = [...(evidenceRefs.get(`${opName}:${varName}`) ?? [])].sort();

    let status;
    if (!fileExists) {
      status = 'missing-wgsl';
    } else if (inlineConfigs.length > 0) {
      status = 'pinned';
    } else if (ruleChains.length > 0 && modelExercisedOps.has(opName)) {
      status = 'model-selectable';
    } else if (ruleChains.length > 0) {
      status = 'selectable';
    } else if (dispatchSources.length > 0) {
      status = 'code-dispatched';
    } else if (evidenceSources.length > 0) {
      status = 'repository-evidence';
    } else {
      status = 'unused';
    }

    statusCounts[status]++;
    // Code and repository evidence stay out of the shipped registry (package
    // byte budget); `--explain <op/variant>` prints them.
    varDef.reachability = { status, inlineConfigs, ruleChains, wgslOnDisk: fileExists };
    if (explainKey === `${opName}/${varName}`) {
      console.log(JSON.stringify({ variant: explainKey, status, inlineConfigs, ruleChains, dispatchSources, evidenceSources }, null, 2));
    }
  }
}

// Companion variants named by a reachable sibling's variantMetadata
// (`reduceVariant`, `normGateVariant`, ...) are dispatched with that sibling.
let companionChanged = true;
while (companionChanged) {
  companionChanged = false;
  for (const [opName, opSchema] of Object.entries(registry.operations ?? {})) {
    for (const [parentName, parentDef] of Object.entries(opSchema.variants ?? {})) {
      if (parentDef.reachability.status === 'unused') continue;
      for (const [key, value] of Object.entries(parentDef.variantMetadata ?? {})) {
        const companion = /Variant$/u.test(key) && typeof value === 'string' ? opSchema.variants[value] : null;
        if (!companion || companion.reachability.status !== 'unused') continue;
        statusCounts.unused--;
        statusCounts['code-dispatched']++;
        companion.reachability.status = 'code-dispatched';
        if (explainKey === `${opName}/${value}`) {
          console.log(`${explainKey}: dispatched with ${opName}/${parentName} (variantMetadata.${key})`);
        }
        companionChanged = true;
      }
    }
  }
}

// Check for WGSL files not in registry at all
const registeredWgsl = new Set();
for (const opSchema of Object.values(registry.operations ?? {})) {
  for (const varDef of Object.values(opSchema.variants ?? {})) {
    if (typeof varDef?.wgsl === 'string') registeredWgsl.add(varDef.wgsl);
  }
}
const unregistered = [...wgslOnDisk].filter((f) => !registeredWgsl.has(f)).sort();
const inventoryIssues = collectInventoryIssues(registry, wgslOnDisk, unregistered);

const content = stringifyRegistry(registry);

if (checkMode) {
  const existing = await fs.readFile(registryPath, 'utf8');
  if (existing === content) {
    console.log(`[kernels:registry:check] registry.json reachability is up to date (${totalVariants} variants)`);
    if (inventoryIssues.length > 0) {
      console.error(`[kernels:registry:check] registry/WGSL inventory has ${inventoryIssues.length} issue(s):`);
      for (const issue of inventoryIssues) {
        console.error(`- ${issue}`);
      }
      process.exit(1);
    }
    process.exit(0);
  }
  console.error('[kernels:registry:check] registry.json reachability is stale — run: npm run kernels:registry:sync');
  if (inventoryIssues.length > 0) {
    console.error(`[kernels:registry:check] registry/WGSL inventory also has ${inventoryIssues.length} issue(s):`);
    for (const issue of inventoryIssues) {
      console.error(`- ${issue}`);
    }
  }
  process.exit(1);
}

await fs.writeFile(registryPath, content);
console.log(`[kernels:registry:sync] wrote reachability for ${totalVariants} variants to registry.json`);
console.log(`  pinned:           ${statusCounts['pinned']}  (execution graph references this exact wgsl+entry)`);
console.log(`  model-selectable: ${statusCounts['model-selectable']}  (rule chain can select; models exercise this operation)`);
console.log(`  selectable:       ${statusCounts['selectable']}  (rule chain can select; no model exercises this operation)`);
console.log(`  code-dispatched:  ${statusCounts['code-dispatched']}  (shipped JS names this variant or its WGSL; no pin or rule)`);
console.log(`  repository-evidence: ${statusCounts['repository-evidence']}  (only repository tests, tools or benchmarks name it)`);
console.log(`  unused:           ${statusCounts['unused']}  (registered, WGSL exists, nothing references it)`);
console.log(`  missing-wgsl:     ${statusCounts['missing-wgsl']}  (registered but WGSL file not on disk)`);
console.log(`  model-exercised operations: ${modelExercisedOps.size}/${registryOpNames.size}`);
if (unregistered.length > 0) {
  console.log(`  unregistered WGSL files (not in registry): ${unregistered.join(', ')}`);
}
if (inventoryIssues.length > 0) {
  console.log(`  registry/WGSL inventory issues: ${inventoryIssues.length}`);
  for (const issue of inventoryIssues) {
    console.log(`    - ${issue}`);
  }
}
