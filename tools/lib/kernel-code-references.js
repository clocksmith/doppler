import fs from 'node:fs/promises';
import path from 'node:path';

// Static evidence that shipped JavaScript can dispatch a registered kernel
// variant without an execution-graph pin or a rule chain. A variant counts as
// code-dispatched when a source file names its WGSL file, names the qualified
// `op/variant`, `op:variant` or `op.variant` key, or names the variant (or a
// template that can produce it) in a file that also names the operation. The
// scan errs toward retention: it proves reachability candidates, never safety
// to delete, and every remaining `unused` entry still needs review.

const EXCLUDED_SOURCE_PREFIXES = [
  'src/config/kernels/',
  'src/gpu/kernels/generated/',
  'src/gpu/kernels/codegen/',
];
const STRING_LITERAL = /'([^'\\\n]*)'|"([^"\\\n]*)"/gu;
const TEMPLATE_LITERAL = /`([^`]*)`/gu;
const TEMPLATE_EXPRESSION = /\$\{[^}]*\}/gu;
const ID_CHARS = '[a-z0-9_]*';

async function walk(dir, predicate, results = []) {
  let items;
  try {
    items = await fs.readdir(dir, { withFileTypes: true });
  } catch (error) {
    if (error?.code === 'ENOENT') return results;
    throw error;
  }
  for (const item of items) {
    const full = path.join(dir, item.name);
    if (item.isDirectory()) await walk(full, predicate, results);
    else if (predicate(full)) results.push(full);
  }
  return results;
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&');
}

function collectLiterals(text) {
  const strings = new Set();
  for (const match of text.matchAll(STRING_LITERAL)) strings.add(match[1] ?? match[2]);
  const templates = [];
  for (const match of text.matchAll(TEMPLATE_LITERAL)) {
    const body = match[1];
    if (!body.includes('${')) {
      strings.add(body);
      continue;
    }
    const parts = body.split(TEMPLATE_EXPRESSION);
    // A template contributes only when it keeps an identifier-shaped literal part;
    // `${a}${b}` alone could name anything and is not evidence.
    if (!parts.some((part) => /[a-z0-9]{2,}/u.test(part))) continue;
    if (!/^[a-z0-9_./${}\s\w]*$/u.test(body)) continue;
    templates.push(new RegExp(`^${parts.map(escapeRegExp).join(ID_CHARS)}$`, 'u'));
  }
  // String concatenation builds ids the same way templates do:
  // `dtype + '_prefill_tile64'` or `'gather_' + suffix`.
  for (const match of text.matchAll(/\+\s*['"](_[a-z0-9_]{4,})['"]/gu)) {
    templates.push(new RegExp(`^${ID_CHARS}${escapeRegExp(match[1])}$`, 'u'));
  }
  for (const match of text.matchAll(/['"]([a-z0-9][a-z0-9_]{2,}_)['"]\s*\+/gu)) {
    templates.push(new RegExp(`^${escapeRegExp(match[1])}${ID_CHARS}$`, 'u'));
  }
  return { strings, templates };
}

function nameMatches(literals, name) {
  return literals.strings.has(name) || literals.templates.some((template) => template.test(name));
}

// Variant ids that name exactly one registered variant and are distinctive
// enough (snake_case with a separator) to identify it without the operation.
function collectDistinctiveVariantIds(registry) {
  const owners = new Map();
  for (const [operation, schema] of Object.entries(registry.operations ?? {})) {
    for (const variant of Object.keys(schema.variants ?? {})) {
      owners.set(variant, owners.has(variant) ? null : operation);
    }
  }
  return new Set([...owners].filter(([variant, op]) => op && variant.includes('_') && variant.length >= 8)
    .map(([variant]) => variant));
}

export async function collectKernelCodeReferences({ repoRoot, registry }) {
  const sourceRoot = path.join(repoRoot, 'src');
  const files = (await walk(sourceRoot, (file) => file.endsWith('.js') && !file.endsWith('.d.js')))
    .map((file) => path.relative(repoRoot, file).split(path.sep).join('/'))
    .filter((file) => !EXCLUDED_SOURCE_PREFIXES.some((prefix) => file.startsWith(prefix)))
    .sort();
  const scanned = [];
  for (const file of files) {
    const text = await fs.readFile(path.join(repoRoot, file), 'utf8');
    scanned.push({ file, literals: collectLiterals(text) });
  }

  const variantsPerWgsl = new Map();
  for (const schema of Object.values(registry.operations ?? {})) {
    for (const definition of Object.values(schema.variants ?? {})) {
      const wgsl = String(definition?.wgsl ?? '');
      variantsPerWgsl.set(wgsl, (variantsPerWgsl.get(wgsl) ?? 0) + 1);
    }
  }
  const distinctive = collectDistinctiveVariantIds(registry);
  const refs = new Map(); // "op:variant" -> Set<file>
  for (const [operation, schema] of Object.entries(registry.operations ?? {})) {
    // Single-variant operations are dispatched by operation name alone
    // (for example backward kernels through the backward executor).
    const singleVariant = Object.keys(schema.variants ?? {}).length === 1;
    for (const [variant, definition] of Object.entries(schema.variants ?? {})) {
      const wgsl = String(definition?.wgsl ?? '');
      const entry = String(definition?.entryPoint ?? 'main');
      // A shared WGSL module identifies one variant only together with its entry.
      const sharedModule = (variantsPerWgsl.get(wgsl) ?? 0) > 1;
      const qualified = [`${operation}/${variant}`, `${operation}:${variant}`, `${operation}.${variant}`];
      const hits = new Set();
      for (const { file, literals } of scanned) {
        const namesWgsl = wgsl
          && (nameMatches(literals, wgsl) || nameMatches(literals, path.posix.basename(wgsl)))
          && (!sharedModule || literals.strings.has(entry));
        const namesQualified = qualified.some((key) => literals.strings.has(key));
        const namesOperation = literals.strings.has(operation);
        const namesVariant = (namesOperation && (singleVariant || nameMatches(literals, variant)))
          || (distinctive.has(variant) && literals.strings.has(variant));
        if (namesWgsl || namesQualified || namesVariant) hits.add(file);
      }
      if (hits.size > 0) refs.set(`${operation}:${variant}`, hits);
    }
  }
  return refs;
}

// Kernel pins declared by Rig recipes under src/config/forge as `[op, variant]`
// pairs (for example the Chronos lowering recipe) or `{ kernel, entry }` refs.
export async function collectForgeRecipeReferences({ repoRoot, registry, collectKernelRefs }) {
  const forgeRoot = path.join(repoRoot, 'src/config/forge');
  const pairRefs = new Map(); // "op:variant" -> Set<recipeId>
  const inlineRefs = new Map(); // "wgsl#entry" -> Set<recipeId>
  const files = await walk(forgeRoot, (file) => file.endsWith('.json'));
  for (const filePath of files.sort()) {
    const recipeId = `forge/${path.relative(forgeRoot, filePath).split(path.sep).join('/').replace(/\.json$/u, '')}`;
    const config = JSON.parse(await fs.readFile(filePath, 'utf8'));
    collectKernelRefs(config, inlineRefs, recipeId);
    const visit = (value) => {
      if (Array.isArray(value)) {
        const [operation, variant] = value;
        if (
          value.length === 2
          && typeof operation === 'string'
          && typeof variant === 'string'
          && registry.operations?.[operation]?.variants?.[variant]
        ) {
          const key = `${operation}:${variant}`;
          if (!pairRefs.has(key)) pairRefs.set(key, new Set());
          pairRefs.get(key).add(recipeId);
        }
        value.forEach(visit);
        return;
      }
      if (value && typeof value === 'object') Object.values(value).forEach(visit);
    };
    visit(config);
  }
  return { pairRefs, inlineRefs };
}

// Repository tests, tools and benchmarks that name a variant's WGSL file or
// variant key. These keep evidence lanes alive but are not shipped dispatch.
export async function collectRepositoryEvidenceReferences({ repoRoot, registry }) {
  const roots = ['tests', 'tools', 'benchmarks'];
  const excluded = ['tests/fixtures/reports/', 'benchmarks/vendors/results/'];
  const files = [];
  for (const root of roots) {
    await walk(path.join(repoRoot, root), (file) => /\.(js|json|html)$/u.test(file), files);
  }
  const scanned = [];
  for (const filePath of files.sort()) {
    const file = path.relative(repoRoot, filePath).split(path.sep).join('/');
    if (excluded.some((prefix) => file.startsWith(prefix))) continue;
    if (file === 'tools/lib/kernel-code-references.js' || file === 'tools/sync-kernel-reachability.js') continue;
    const text = await fs.readFile(filePath, 'utf8');
    scanned.push({ file, text, literals: collectLiterals(text) });
  }
  const distinctive = collectDistinctiveVariantIds(registry);
  const refs = new Map();
  for (const [operation, schema] of Object.entries(registry.operations ?? {})) {
    for (const [variant, definition] of Object.entries(schema.variants ?? {})) {
      const base = path.posix.basename(String(definition?.wgsl ?? ''));
      const basePattern = new RegExp(`(^|[^a-z0-9_])${escapeRegExp(base)}`, 'u');
      const qualified = [`${operation}/${variant}`, `${operation}:${variant}`, `${operation}.${variant}`];
      const hits = new Set();
      for (const { file, text, literals } of scanned) {
        if (
          (base && basePattern.test(text))
          || qualified.some((key) => literals.strings.has(key))
          || (literals.strings.has(operation) && nameMatches(literals, variant))
          || (distinctive.has(variant) && nameMatches(literals, variant))
        ) {
          hits.add(file);
        }
      }
      if (hits.size > 0) refs.set(`${operation}:${variant}`, hits);
    }
  }
  return refs;
}
