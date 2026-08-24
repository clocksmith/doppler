#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const policyPath = path.join(repoRoot, 'tools/policies/source-architecture-policy.json');
const importPattern = /\b(?:import|export)\s+(?:[^'\"]*?\sfrom\s*)?['\"]([^'\"]+)['\"]/g;
const dynamicImportPattern = /\bimport\s*\(\s*['\"]([^'\"]+)['\"]\s*\)/g;
const facadeImplementationPattern = /^\s*(?:export\s+)?(?:async\s+)?(?:function|class|const|let|var)\b/m;
const genericModulePattern = /(?:^|\/)(?:utils|helpers)\.js$|-(?:utils|helpers|shared)\.js$|\.shared\.js$|(?:^|\/)shared-runtime\.schema\.js$/;
const governedExtensions = new Set(['.js', '.d.ts', '.wgsl']);
const genericModuleClassifications = new Set([
  'semantic-owner',
  'compatibility-facade',
  'legacy-debt',
]);

function toPosix(value) {
  return value.split(path.sep).join('/');
}

function countLines(source) {
  if (source.length === 0) return 0;
  const lines = source.split(/\r?\n/);
  return lines.at(-1) === '' ? lines.length - 1 : lines.length;
}

async function walk(directory) {
  const files = [];
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await walk(entryPath));
    else if (entry.isFile()) files.push(entryPath);
  }
  return files;
}

function collectSpecifiers(source) {
  const values = [];
  for (const pattern of [importPattern, dynamicImportPattern]) {
    pattern.lastIndex = 0;
    for (;;) {
      const match = pattern.exec(source);
      if (!match) break;
      values.push(match[1]);
    }
  }
  return values;
}

function ownerFor(sourceRoot, targetPath) {
  const relative = path.relative(sourceRoot, targetPath);
  if (relative.startsWith('..') || path.isAbsolute(relative)) return null;
  return relative.split(path.sep)[0];
}

function declaredOwnerFor(policy, sourceRoot, targetPath) {
  const relative = toPosix(path.relative(sourceRoot, targetPath));
  if (relative.startsWith('..') || path.isAbsolute(relative)) return null;
  if (!relative.includes('/')) {
    const stem = relative.replace(/\.d\.ts$|\.[^.]+$/u, '');
    return policy.rootFileOwners?.[stem] ?? null;
  }
  return ownerFor(sourceRoot, targetPath);
}

function exceptionKey(from, toOwner) {
  return `${from}->${toOwner}`;
}

function resolveSourceImport(filePath, specifier, sourceFiles) {
  if (!specifier.startsWith('.')) return null;
  const target = path.resolve(path.dirname(filePath), specifier);
  for (const candidate of [target, `${target}.js`, path.join(target, 'index.js')]) {
    if (sourceFiles.has(candidate)) return candidate;
  }
  return null;
}

async function relativeImportExists(filePath, specifier) {
  if (!specifier.startsWith('.')) return true;
  const target = path.resolve(path.dirname(filePath), specifier);
  const candidates = [target];
  if (!path.extname(target)) {
    candidates.push(`${target}.js`, `${target}.json`, path.join(target, 'index.js'));
  }
  for (const candidate of candidates) {
    const stat = await fs.stat(candidate).catch(() => null);
    if (stat?.isFile()) return true;
  }
  return false;
}

function collectStronglyConnectedComponents(graph) {
  let nextIndex = 0;
  const indices = new Map();
  const lowLinks = new Map();
  const stack = [];
  const onStack = new Set();
  const components = [];

  function visit(node) {
    indices.set(node, nextIndex);
    lowLinks.set(node, nextIndex);
    nextIndex += 1;
    stack.push(node);
    onStack.add(node);

    for (const dependency of graph.get(node) ?? []) {
      if (!graph.has(dependency)) continue;
      if (!indices.has(dependency)) {
        visit(dependency);
        lowLinks.set(node, Math.min(lowLinks.get(node), lowLinks.get(dependency)));
      } else if (onStack.has(dependency)) {
        lowLinks.set(node, Math.min(lowLinks.get(node), indices.get(dependency)));
      }
    }

    if (lowLinks.get(node) !== indices.get(node)) return;
    const component = [];
    for (;;) {
      const current = stack.pop();
      onStack.delete(current);
      component.push(current);
      if (current === node) break;
    }
    if (component.length > 1 || (graph.get(node) ?? []).includes(node)) {
      components.push(component);
    }
  }

  for (const node of graph.keys()) {
    if (!indices.has(node)) visit(node);
  }
  return components;
}

function collectPackageSourceEntryPoints(packageJson) {
  const entries = [];
  function collect(value) {
    if (typeof value === 'string') {
      if (value.startsWith('./src/') && value.endsWith('.js')) entries.push(value.slice('./src/'.length));
      else if (value.startsWith('src/') && value.endsWith('.js')) entries.push(value.slice('src/'.length));
      return;
    }
    if (!value || typeof value !== 'object') return;
    for (const child of Object.values(value)) collect(child);
  }
  collect(packageJson.exports);
  collect(packageJson.bin);
  return entries;
}

async function collectExternalSourceRoots(policy, sourceRoot, sourceFiles) {
  const roots = new Set();
  for (const relativeRoot of policy.reachability?.externalRoots ?? []) {
    const externalRoot = path.join(repoRoot, relativeRoot);
    const stat = await fs.stat(externalRoot).catch(() => null);
    if (!stat?.isDirectory()) continue;
    const externalFiles = await walk(externalRoot);
    for (const filePath of externalFiles) {
      if (path.extname(filePath) !== '.js') continue;
      const source = await fs.readFile(filePath, 'utf8');
      for (const specifier of collectSpecifiers(source)) {
        const imported = resolveSourceImport(filePath, specifier, sourceFiles);
        if (imported) roots.add(imported);
      }
    }
  }
  return roots;
}

async function validateProductionGraph(policy, sourceRoot, files, facadePaths, errors) {
  const sourceFiles = new Set(files.filter((filePath) => path.extname(filePath) === '.js'));
  const fullGraph = new Map();
  const cycleGraph = new Map();
  const experimentalBridges = new Map(Object.entries(policy.experimentalBridges ?? {}));
  const observedExperimentalBridges = new Set();

  for (const filePath of sourceFiles) {
    const relative = toPosix(path.relative(sourceRoot, filePath));
    const source = await fs.readFile(filePath, 'utf8');
    const dependencies = [];
    for (const specifier of collectSpecifiers(source)) {
      if (!specifier.startsWith('.')) continue;
      const imported = resolveSourceImport(filePath, specifier, sourceFiles);
      if (!imported) {
        if (!await relativeImportExists(filePath, specifier)) {
          errors.push(`${relative}: unresolved relative import ${specifier}`);
        }
        continue;
      }
      dependencies.push(imported);

      const fromOwner = declaredOwnerFor(policy, sourceRoot, filePath);
      const toOwner = declaredOwnerFor(policy, sourceRoot, imported);
      if (fromOwner !== 'experimental' && toOwner === 'experimental' && !facadePaths.has(relative)) {
        const bridge = experimentalBridges.get(relative);
        if (!bridge || typeof bridge.reason !== 'string' || !bridge.reason.trim()) {
          errors.push(`${relative}: production module imports experimental code without a declared quarantine bridge`);
        } else {
          observedExperimentalBridges.add(relative);
        }
      }
    }
    fullGraph.set(filePath, [...new Set(dependencies)]);
    if (!facadePaths.has(relative)) {
      cycleGraph.set(
        filePath,
        [...new Set(dependencies)].filter((dependency) => {
          const dependencyRelative = toPosix(path.relative(sourceRoot, dependency));
          return !facadePaths.has(dependencyRelative);
        })
      );
    }
  }

  for (const relative of experimentalBridges.keys()) {
    if (!observedExperimentalBridges.has(relative)) {
      errors.push(`experimental quarantine bridge is stale: ${relative}`);
    }
  }

  for (const component of collectStronglyConnectedComponents(cycleGraph)) {
    const members = component
      .map((filePath) => toPosix(path.relative(sourceRoot, filePath)))
      .sort();
    errors.push(`production import cycle: ${members.join(' -> ')}`);
  }

  const packageJson = JSON.parse(await fs.readFile(path.join(repoRoot, 'package.json'), 'utf8'));
  const entryPaths = new Set();
  for (const relative of [
    ...collectPackageSourceEntryPoints(packageJson),
    ...(policy.reachability?.roots ?? []),
    ...Object.keys(policy.reachability?.standaloneModules ?? {}),
  ]) {
    const entryPath = path.join(sourceRoot, relative);
    if (!sourceFiles.has(entryPath)) errors.push(`source reachability root is missing: ${relative}`);
    else entryPaths.add(entryPath);
  }
  for (const entryPath of await collectExternalSourceRoots(policy, sourceRoot, sourceFiles)) {
    entryPaths.add(entryPath);
  }

  for (const [relative, review] of Object.entries(policy.reachability?.standaloneModules ?? {})) {
    if (typeof review.reason !== 'string' || !review.reason.trim()) {
      errors.push(`standalone source module lacks reason: ${relative}`);
    }
  }

  const reachable = new Set();
  const pending = [...entryPaths];
  while (pending.length > 0) {
    const filePath = pending.pop();
    if (reachable.has(filePath)) continue;
    reachable.add(filePath);
    for (const dependency of fullGraph.get(filePath) ?? []) pending.push(dependency);
  }
  for (const filePath of sourceFiles) {
    const relative = toPosix(path.relative(sourceRoot, filePath));
    const owner = declaredOwnerFor(policy, sourceRoot, filePath);
    if (owner === 'experimental' || facadePaths.has(relative)) continue;
    if (!reachable.has(filePath)) errors.push(`${relative}: unreachable production module`);
  }
}

async function validateConstitutionalDomains(policy, sourceRoot, files, errors) {
  const sourceFiles = new Set(files.filter((filePath) => path.extname(filePath) === '.js'));
  const domainFiles = new Map();
  for (const [domain, relativePaths] of Object.entries(policy.constitutionalDomains ?? {})) {
    const resolved = new Set();
    for (const relative of relativePaths) {
      const filePath = path.join(sourceRoot, relative);
      if (!sourceFiles.has(filePath)) errors.push(`constitutional ${domain} owner is missing: ${relative}`);
      else resolved.add(filePath);
    }
    domainFiles.set(domain, resolved);
  }
  for (const rule of policy.constitutionalImportGraphs ?? []) {
    if (!domainFiles.has(rule.domain)) {
      errors.push(`constitutional import graph references unknown domain: ${rule.domain}`);
      continue;
    }
    const pending = [];
    for (const relative of rule.entryPoints ?? []) {
      const entryPath = path.join(sourceRoot, relative);
      if (!sourceFiles.has(entryPath)) errors.push(`constitutional ${rule.domain} entry point is missing: ${relative}`);
      else pending.push(entryPath);
    }
    const visited = new Set();
    while (pending.length > 0) {
      const filePath = pending.pop();
      if (visited.has(filePath)) continue;
      visited.add(filePath);
      const relative = toPosix(path.relative(sourceRoot, filePath));
      for (const prefix of rule.forbiddenPathPrefixes ?? []) {
        if (relative.startsWith(prefix)) {
          errors.push(`constitutional ${rule.domain} import graph reaches forbidden path ${relative}`);
        }
      }
      const source = await fs.readFile(filePath, 'utf8');
      for (const specifier of collectSpecifiers(source)) {
        const imported = resolveSourceImport(filePath, specifier, sourceFiles);
        if (imported && !visited.has(imported)) pending.push(imported);
      }
    }
  }
}

async function main() {
  const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
  const sourceRoot = path.join(repoRoot, policy.sourceRoot);
  const errors = [];
  const actualOwners = (await fs.readdir(sourceRoot, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
  const expectedOwners = Object.keys(policy.topLevelOwners).sort();
  if (JSON.stringify(actualOwners) !== JSON.stringify(expectedOwners)) {
    errors.push(`top-level source owners differ: expected=${expectedOwners.join(',')} actual=${actualOwners.join(',')}`);
  }

  const exceptions = new Map();
  for (const entry of policy.dependencyExceptions) {
    if (!entry.reason || !entry.extractTo) {
      errors.push(`dependency exception lacks reason or extractTo: ${JSON.stringify(entry)}`);
      continue;
    }
    const key = exceptionKey(entry.from, entry.toOwner);
    if (exceptions.has(key)) errors.push(`duplicate dependency exception: ${key}`);
    exceptions.set(key, { ...entry, used: false });
  }

  const legacyOversize = new Map(Object.entries(policy.legacyOversize));
  const observedLegacy = new Set();
  const softLimitReviews = new Map(Object.entries(policy.softLimitReviews ?? {}));
  const observedSoftLimitReviews = new Set();
  const genericModuleReviews = new Map(Object.entries(policy.genericModuleReviews ?? {}));
  const observedGenericModules = new Set();
  const facadePaths = new Set(policy.facades ?? []);
  const files = await walk(sourceRoot);
  await validateConstitutionalDomains(policy, sourceRoot, files, errors);
  await validateProductionGraph(policy, sourceRoot, files, facadePaths, errors);
  for (const filePath of files) {
    const relative = toPosix(path.relative(sourceRoot, filePath));
    const source = await fs.readFile(filePath, 'utf8');
    const extension = relative.endsWith('.d.ts') ? '.d.ts' : path.extname(relative);
    const declaredOwner = governedExtensions.has(extension)
      ? declaredOwnerFor(policy, sourceRoot, filePath)
      : null;
    if (governedExtensions.has(extension) && !declaredOwner) {
      errors.push(`${relative}: governed source file has no declared owner`);
    } else if (declaredOwner && !Object.hasOwn(policy.topLevelOwners, declaredOwner)) {
      errors.push(`${relative}: declared owner "${declaredOwner}" is not a top-level owner`);
    }
    if (governedExtensions.has(extension)) {
      const lineCount = countLines(source);
      const legacy = legacyOversize.get(relative);
      if (legacy) {
        observedLegacy.add(relative);
        if (!Array.isArray(legacy.extractTo) || legacy.extractTo.length === 0) {
          errors.push(`${relative}: legacy oversize entry lacks extraction boundaries`);
        }
        if (lineCount > legacy.maxLines) {
          errors.push(`${relative}: grew from governed ceiling ${legacy.maxLines} to ${lineCount} lines`);
        }
        if (lineCount <= policy.lineLimit) {
          errors.push(`${relative}: now satisfies ${policy.lineLimit}; remove stale legacyOversize entry`);
        }
      } else if (lineCount > policy.lineLimit) {
        errors.push(`${relative}: ${lineCount} lines exceeds blocking limit ${policy.lineLimit}`);
      }
      const softReview = softLimitReviews.get(relative);
      if (!legacy && lineCount > policy.softLineLimit && lineCount <= policy.lineLimit) {
        observedSoftLimitReviews.add(relative);
        if (!softReview) {
          errors.push(`${relative}: ${lineCount} lines exceeds soft limit ${policy.softLineLimit} without review`);
        } else {
          if (!Number.isInteger(softReview.reviewedLines) || softReview.reviewedLines < lineCount) {
            errors.push(`${relative}: grew beyond reviewed soft-limit ceiling ${String(softReview.reviewedLines)} to ${lineCount}`);
          }
          if (typeof softReview.reason !== 'string' || !softReview.reason.trim()) {
            errors.push(`${relative}: soft-limit review lacks reason`);
          }
          if (!Array.isArray(softReview.extractTo) || softReview.extractTo.length === 0) {
            errors.push(`${relative}: soft-limit review lacks a named extraction boundary`);
          }
        }
      }
    }

    if (path.extname(relative) !== '.js') continue;
    if (genericModulePattern.test(relative)) observedGenericModules.add(relative);
    if (facadePaths.has(relative)) continue;
    const fromOwner = declaredOwnerFor(policy, sourceRoot, filePath);
    const allowedOwners = policy.allowedDependencies?.[fromOwner];
    if (!Array.isArray(allowedOwners)) {
      errors.push(`${relative}: owner ${String(fromOwner)} lacks an allowed dependency policy`);
      continue;
    }
    for (const specifier of collectSpecifiers(source)) {
      if (!specifier.startsWith('.')) continue;
      const sourceFiles = new Set(files.filter((candidate) => path.extname(candidate) === '.js'));
      const imported = resolveSourceImport(filePath, specifier, sourceFiles);
      if (!imported) continue;
      const toOwner = declaredOwnerFor(policy, sourceRoot, imported);
      if (!toOwner || allowedOwners.includes(toOwner)) continue;
      const key = exceptionKey(relative, toOwner);
      const exception = exceptions.get(key);
      if (exception) {
        exception.used = true;
        continue;
      }
      errors.push(`${relative}: dependency on ${toOwner} violates ${fromOwner} ownership boundary`);
    }
  }

  for (const relative of legacyOversize.keys()) {
    if (!observedLegacy.has(relative)) errors.push(`legacyOversize entry is stale or missing: ${relative}`);
  }
  for (const relative of softLimitReviews.keys()) {
    if (!observedSoftLimitReviews.has(relative)) {
      errors.push(`softLimitReviews entry is stale or below the soft limit: ${relative}`);
    }
  }
  for (const [key, entry] of exceptions) {
    if (!entry.used) errors.push(`dependency exception is stale: ${key}`);
  }
  for (const relative of [...observedGenericModules].sort()) {
    const review = genericModuleReviews.get(relative);
    if (!review) {
      errors.push(`${relative}: generic module lacks a semantic ownership review`);
      continue;
    }
    if (!genericModuleClassifications.has(review.classification)) {
      errors.push(`${relative}: generic module has invalid classification "${String(review.classification)}"`);
    }
    if (typeof review.semanticOwner !== 'string' || !review.semanticOwner.trim()) {
      errors.push(`${relative}: generic module review lacks semanticOwner`);
    }
    if (review.classification === 'compatibility-facade' && !facadePaths.has(relative)) {
      errors.push(`${relative}: compatibility facade is missing from policy.facades`);
    }
    if (
      review.classification === 'legacy-debt'
      && (!Array.isArray(review.extractTo) || review.extractTo.length === 0)
    ) {
      errors.push(`${relative}: generic legacy debt lacks extraction boundaries`);
    }
  }
  for (const relative of genericModuleReviews.keys()) {
    if (!observedGenericModules.has(relative)) {
      errors.push(`genericModuleReviews entry is stale or missing: ${relative}`);
    }
  }
  for (const relative of policy.facades) {
    const facadePath = path.join(sourceRoot, relative);
    const source = await fs.readFile(facadePath, 'utf8').catch(() => null);
    if (source === null) {
      errors.push(`declared facade is missing: ${relative}`);
      continue;
    }
    if (facadeImplementationPattern.test(source)) {
      errors.push(`${relative}: facade contains an implementation declaration`);
    }
  }
  for (const owner of policy.compatibilityOnlyOwners ?? []) {
    if (!Object.hasOwn(policy.topLevelOwners, owner)) {
      errors.push(`compatibility-only owner is unknown: ${owner}`);
      continue;
    }
    const ownerRoot = path.join(sourceRoot, owner);
    const ownerFiles = await walk(ownerRoot).catch(() => []);
    for (const filePath of ownerFiles) {
      if (path.extname(filePath) !== '.js') continue;
      const relative = toPosix(path.relative(sourceRoot, filePath));
      if (!facadePaths.has(relative)) {
        errors.push(`${relative}: compatibility-only owner ${owner} contains an undeclared implementation`);
      }
    }
  }

  if (errors.length > 0) {
    console.error('source architecture check failed:');
    for (const error of errors) console.error(`- ${error}`);
    process.exitCode = 1;
    return;
  }
  const genericCounts = { semanticOwner: 0, compatibilityFacade: 0, legacyDebt: 0 };
  for (const relative of observedGenericModules) {
    const classification = genericModuleReviews.get(relative)?.classification;
    if (classification === 'semantic-owner') genericCounts.semanticOwner += 1;
    if (classification === 'compatibility-facade') genericCounts.compatibilityFacade += 1;
    if (classification === 'legacy-debt') genericCounts.legacyDebt += 1;
  }
  console.log(
    `source architecture check passed: ${actualOwners.length} owners, ${files.length} files, `
    + `${legacyOversize.size} governed oversize files, ${observedSoftLimitReviews.size} reviewed soft-limit files, `
    + `${observedGenericModules.size} reviewed generic modules `
    + `(${genericCounts.semanticOwner} semantic owners, ${genericCounts.compatibilityFacade} facades, `
    + `${genericCounts.legacyDebt} legacy debt)`
  );
}

await main();
