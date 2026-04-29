#!/usr/bin/env node

/**
 * Computes and writes `reachability` metadata into each variant in
 * src/config/kernels/registry.json.
 *
 * For every variant the tool resolves:
 *   - Which conversion configs inline it (via execution.kernels)
 *   - Which rule chains can select it (via src/rules/kernels/*.rules.json)
 *   - Whether the WGSL file exists on disk
 *   - A status classification: pinned, model-selectable, selectable, unused, missing-wgsl
 *
 * Usage:
 *   npm run kernels:registry:sync
 *   npm run kernels:registry:check
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const registryPath = path.join(repoRoot, 'src/config/kernels/registry.json');
const kernelsDir = path.join(repoRoot, 'src/gpu/kernels');
const rulesDir = path.join(repoRoot, 'src/rules/kernels');
const conversionRoot = path.join(repoRoot, 'src/config/conversion');

const checkMode = process.argv.includes('--check');
const ID_PATTERN = /^[a-z0-9]+(?:_[a-z0-9]+)*$/u;
const WGSL_SEGMENT_PATTERN = /^[a-z0-9]+(?:_[a-z0-9]+)*(?:\.wgsl)?$/u;

// ---------------------------------------------------------------------------
// Pass 1: Collect inline references from conversion configs
// ---------------------------------------------------------------------------

async function collectInlineRefs() {
  const refs = new Map(); // key = "wgsl#entry" -> Set<configId>
  const entries = await walkJsonFiles(conversionRoot);
  for (const filePath of entries) {
    const config = JSON.parse(await fs.readFile(filePath, 'utf8'));
    const kernels = config?.execution?.kernels;
    if (!kernels || typeof kernels !== 'object') continue;
    const configId = path.relative(conversionRoot, filePath).replace(/\.json$/, '');
    for (const kDef of Object.values(kernels)) {
      const wgsl = kDef?.kernel;
      const entry = kDef?.entry ?? 'main';
      if (typeof wgsl !== 'string') continue;
      const key = `${wgsl}#${entry}`;
      if (!refs.has(key)) refs.set(key, new Set());
      refs.get(key).add(configId);
    }
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

async function collectRuleRefs(registryOpNames) {
  const refs = new Map(); // key = "operation:variantName" -> Set<"file#chain">
  const ruleFiles = (await fs.readdir(rulesDir))
    .filter((f) => f.endsWith('.rules.json'));

  for (const file of ruleFiles) {
    // Infer target operation(s) from filename: kebab-to-snake
    const ruleBase = file.replace('.rules.json', '').replace(/-/g, '_');
    const targetOps = registryOpNames.has(ruleBase)
      ? [ruleBase]
      : [...registryOpNames].filter((op) => op.startsWith(ruleBase + '_'));

    if (targetOps.length === 0) continue;

    // Collect raw variant names from this rule file
    const rawRefs = new Map(); // variantName -> Set<source>
    const rules = JSON.parse(await fs.readFile(path.join(rulesDir, file), 'utf8'));
    for (const [chainName, chain] of Object.entries(rules)) {
      if (!Array.isArray(chain)) continue;
      for (const rule of chain) {
        extractVariantNames(rule?.value, rawRefs, `${file}#${chainName}`);
      }
    }

    // Scope each variant name to target operations
    for (const [varName, sources] of rawRefs) {
      for (const op of targetOps) {
        const key = `${op}:${varName}`;
        if (!refs.has(key)) refs.set(key, new Set());
        for (const s of sources) refs.get(key).add(s);
      }
    }
  }
  return refs;
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
const inlineRefs = await collectInlineRefs();
const ruleRefs = await collectRuleRefs(registryOpNames);
const wgslOnDisk = await collectWgslOnDisk();
const reverseIndex = buildReverseIndex(registry);
const modelExercisedOps = collectModelExercisedOps(inlineRefs, reverseIndex);

let totalVariants = 0;
const statusCounts = {
  'pinned': 0,
  'model-selectable': 0,
  'selectable': 0,
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

    const inlineConfigs = [...inlineConfigSet].sort();
    const ruleChains = [...ruleRefSet].sort();

    let status;
    if (!fileExists) {
      status = 'missing-wgsl';
    } else if (inlineConfigs.length > 0) {
      status = 'pinned';
    } else if (ruleChains.length > 0 && modelExercisedOps.has(opName)) {
      status = 'model-selectable';
    } else if (ruleChains.length > 0) {
      status = 'selectable';
    } else {
      status = 'unused';
    }

    statusCounts[status]++;
    varDef.reachability = { status, inlineConfigs, ruleChains, wgslOnDisk: fileExists };
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

const content = JSON.stringify(registry, null, 2) + '\n';

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
