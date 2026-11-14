#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { checkJavaScriptSource, RULE_IDS as JS_RULES } from './check-javascript-style.js';
import { checkJsonSource, RULE_IDS as JSON_RULES } from './check-json-style.js';
import { checkWgslSource, RULE_IDS as WGSL_RULES } from './check-wgsl-style.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const EXTENSIONS = new Set(['.js', '.mjs', '.json', '.wgsl', '.wgslinc']);

export function listStyleFiles(root = ROOT) {
  return [...new Set(execFileSync('git', ['ls-files', '-z', '--cached', '--others', '--exclude-standard'], {
    cwd: root, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024,
  }).split('\0').filter(file => file && EXTENSIONS.has(path.extname(file)) && fs.existsSync(path.join(root, file))))].sort();
}

export function validateStyleRegistry(registry, root = ROOT) {
  const errors = [];
  const ids = new Set();
  for (const rule of registry.rules) {
    if (ids.has(rule.id)) errors.push(`Duplicate rule ID ${rule.id}`);
    ids.add(rule.id);
    if (!registry.guides.includes(rule.guide)) errors.push(`${rule.id}: unknown guide ${rule.guide}`);
    if (!['baseline', 'scoped', 'review'].includes(rule.coverage)) errors.push(`${rule.id}: unknown coverage`);
    if (!rule.limits) errors.push(`${rule.id}: missing scope/limitations`);
    if (rule.coverage === 'baseline' && (!rule.checker || !rule.fixture)) errors.push(`${rule.id}: baseline rules require detector and positive/negative fixtures`);
    for (const file of [rule.checker, rule.fixture].filter(Boolean)) {
      if (!fs.existsSync(path.join(root, file))) errors.push(`${rule.id}: missing ${file}`);
    }
  }
  for (const id of [...JS_RULES, ...JSON_RULES, ...WGSL_RULES]) {
    if (!ids.has(id)) errors.push(`Unregistered detector rule ${id}`);
  }
  const guideDirectory = path.join(root, 'docs/style');
  for (const file of fs.readdirSync(guideDirectory).filter(file => file.endsWith('.md') && file !== 'README.md')) {
    if (!registry.guides.includes(`docs/style/${file}`)) errors.push(`Unregistered guide ${file}`);
  }
  for (const guide of registry.guides) {
    const source = fs.readFileSync(path.join(root, guide), 'utf8');
    const expected = registry.rules.filter(rule => rule.guide === guide).map(rule => rule.id);
    if (!expected.length) errors.push(`${guide}: no rule coverage registered`);
    const marker = `<!-- style-rules: ${expected.join(', ')} -->`;
    if (!source.includes(marker)) errors.push(`${guide}: missing or stale rule marker: ${marker}`);
  }
  const legacy = new Set();
  for (const entry of registry.legacyMjs) {
    if (legacy.has(entry.path) || !entry.path.endsWith('.mjs') || !entry.reason || !entry.removalCondition || !fs.existsSync(path.join(root, entry.path))) {
      errors.push(`Invalid, duplicate or stale legacy .mjs record ${entry.path}`);
    }
    legacy.add(entry.path);
  }
  return errors;
}

export function checkStyleFile(source, file, registry) {
  const extension = path.extname(file);
  if (extension === '.json') return checkJsonSource(source, file);
  if (extension === '.wgsl' || extension === '.wgslinc') return checkWgslSource(source, file);
  return checkJavaScriptSource(source, file, registry.legacyMjs);
}

export function diagnosticKey(file, diagnostic, source) {
  return JSON.stringify([file, diagnostic.ruleId, source.split(/\r?\n/)[(diagnostic.line ?? 1) - 1]?.trim() ?? '']);
}

export function changedStyleFiles(root = ROOT) {
  const files = new Set(execFileSync('git', ['diff', '--name-only', '-z', 'HEAD'], { cwd: root, encoding: 'utf8' }).split('\0'));
  for (const file of execFileSync('git', ['ls-files', '-z', '--others', '--exclude-standard'], { cwd: root, encoding: 'utf8' }).split('\0')) files.add(file);
  // Include the most recent committed patch as well as staged/unstaged/new work.
  for (const file of execFileSync('git', ['diff-tree', '--root', '--no-commit-id', '--name-only', '-r', '-z', 'HEAD'], { cwd: root, encoding: 'utf8' }).split('\0')) files.add(file);
  return files;
}

export function runStyleGuides(root = ROOT, changedOnly = false) {
  const registry = JSON.parse(fs.readFileSync(path.join(root, 'tools/policies/style-guide-rules.json'), 'utf8'));
  const errors = validateStyleRegistry(registry, root);
  const debt = JSON.parse(fs.readFileSync(path.join(root, 'tools/policies/style-guide-debt.json'), 'utf8'));
  const allowances = new Map();
  for (const entry of debt.entries) {
    const key = JSON.stringify([entry.file, entry.ruleId, entry.sourceLine]);
    if (!registry.rules.some(rule => rule.id === entry.ruleId) || typeof entry.file !== "string" || typeof entry.sourceLine !== "string") errors.push(`Malformed debt record ${key}`);
    if (!entry.reason || !entry.removalCondition || !Number.isInteger(entry.count) || entry.count < 1 || allowances.has(key)) {
      errors.push(`Invalid duplicate/debt record ${key}`);
    }
    allowances.set(key, entry.count);
  }
  const counts = {};
  const inventory = listStyleFiles(root);
  const changed = changedOnly ? changedStyleFiles(root) : null;
  const selected = new Set(inventory.filter(file => !changed || changed.has(file)));
  let retained = 0;
  for (const file of selected) {
    const source = fs.readFileSync(path.join(root, file), 'utf8');
    const extension = path.extname(file);
    counts[extension] = (counts[extension] ?? 0) + 1;
    for (const diagnostic of checkStyleFile(source, file, registry)) {
      const key = diagnosticKey(file, diagnostic, source);
      const remaining = allowances.get(key) ?? 0;
      if (remaining > 0) { allowances.set(key, remaining - 1); retained++; }
      else errors.push(`${file}:${diagnostic.line ?? 1} [${diagnostic.ruleId}] ${diagnostic.message}`);
    }
  }
  for (const [key, count] of allowances) if (count > 0 && (!changedOnly || selected.has(JSON.parse(key)[0]) || !fs.existsSync(path.join(root, JSON.parse(key)[0])))) errors.push(`Stale style debt (${count} unmatched): ${key}`);
  return { errors, counts, retained, inventoried: inventory.length, checked: selected.size };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.slice(2).some(arg => arg !== '--changed')) throw new Error('Usage: node tools/check-style-guides.js [--changed]');
  const changedOnly = process.argv.includes('--changed');
  const result = runStyleGuides(ROOT, changedOnly);
  console.log(`[style:guides] ${changedOnly ? 'changed-file' : 'full-file'} checks ${result.checked}/${result.inventoried} inventoried ${JSON.stringify(result.counts)}; ${result.retained} explicit retained violations; baseline coverage only, scoped and semantic rules remain separate`);
  if (result.errors.length) {
    console.error(result.errors.join('\n'));
    process.exitCode = 1;
  }
}
