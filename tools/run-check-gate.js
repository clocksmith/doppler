#!/usr/bin/env node

// Repository check gate: always-on behavior checks plus input-triggered
// mirror, drift and policy checks. Policy: tools/policies/check-gate-policy.json.
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { moduleSpecifiers } from './lib/javascript-dependency-graph.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const POLICY_PATH = 'tools/policies/check-gate-policy.json';
const GATE_FILES = [POLICY_PATH, 'tools/run-check-gate.js'];
const USAGE = 'Usage: node tools/run-check-gate.js [--behavior] [--changed [--base <ref>]] [--all] [--list]';

export function globToRegExp(glob) {
  let pattern = '';
  for (let i = 0; i < glob.length; i += 1) {
    const char = glob[i];
    if (char === '*' && glob[i + 1] === '*') {
      const slash = glob[i + 2] === '/';
      pattern += slash ? '(?:.*/)?' : '.*';
      i += slash ? 2 : 1;
    } else if (char === '*') pattern += '[^/]*';
    else if (char === '?') pattern += '[^/]';
    else pattern += /[.+^${}()|[\]\\]/.test(char) ? `\\${char}` : char;
  }
  return new RegExp(`^${pattern}$`);
}

export function validatePolicy(policy, scripts) {
  const errors = [];
  if (policy?.schemaVersion !== 1) errors.push('schemaVersion must be 1');
  if (!Array.isArray(policy?.behavior) || policy.behavior.length === 0) errors.push('behavior must be a non-empty array');
  if (!policy?.triggered || typeof policy.triggered !== 'object' || Array.isArray(policy.triggered)) {
    errors.push('triggered must be an object');
  }
  if (errors.length) return errors;
  for (const name of policy.behavior) {
    if (!Object.hasOwn(scripts, name)) errors.push(`behavior script "${name}" is not declared in package.json`);
    if (Object.hasOwn(policy.triggered, name)) errors.push(`script "${name}" is both behavior and triggered`);
  }
  for (const [name, entry] of Object.entries(policy.triggered)) {
    if (!Object.hasOwn(scripts, name)) errors.push(`triggered script "${name}" is not declared in package.json`);
    const unknown = Object.keys(entry ?? {}).filter((key) => !['inputs', 'referencedBy'].includes(key));
    if (unknown.length) errors.push(`${name}: unknown fields ${unknown.join(', ')}`);
    if (!Array.isArray(entry?.inputs) || entry.inputs.length === 0
      || entry.inputs.some((glob) => typeof glob !== 'string' || glob === '' || glob.startsWith('/'))) {
      errors.push(`${name}: inputs must be a non-empty array of repo-relative globs`);
    }
    if (entry?.referencedBy !== undefined && (!Array.isArray(entry.referencedBy)
      || entry.referencedBy.some((file) => typeof file !== 'string' || file.startsWith('/')))) {
      errors.push(`${name}: referencedBy must be an array of repo-relative JSON paths`);
    }
  }
  return errors;
}

function toRepoPath(root, absolute) {
  return path.relative(root, absolute).split(path.sep).join('/');
}

// Files a script's commands execute: tool and test entry points, tsconfig
// files, and nested npm scripts. Their local import closure is added below.
export function scriptEntryFiles(name, scripts, root = REPO_ROOT, seen = new Set()) {
  if (seen.has(name)) return [];
  seen.add(name);
  const entries = [];
  for (const command of String(scripts[name] ?? '').split(/\s*&&\s*/)) {
    const tokens = command.trim().split(/\s+/);
    if (tokens[0] === 'npm' && tokens[1] === 'run' && tokens[2]) {
      entries.push(...scriptEntryFiles(tokens[2], scripts, root, seen));
      continue;
    }
    for (let i = 1; i < tokens.length; i += 1) {
      const token = tokens[i].replace(/^['"]|['"]$/g, '');
      if (tokens[0] === 'tsc' && tokens[i - 1] === '-p') entries.push(token);
      else if (/\.(?:js|json)$/.test(token) && !token.includes('*') && existsSync(path.join(root, token))) entries.push(token);
      else if (/^tests\//.test(token) && existsSync(path.join(root, token)) && statSync(path.join(root, token)).isFile()) entries.push(token);
    }
  }
  return entries;
}

export function localImportClosure(entries, root = REPO_ROOT) {
  const closure = new Set();
  const pending = [...entries];
  while (pending.length) {
    const file = pending.pop();
    if (closure.has(file)) continue;
    const absolute = path.join(root, file);
    if (!existsSync(absolute) || !statSync(absolute).isFile()) continue;
    closure.add(file);
    if (!/\.(?:js|cjs)$/.test(file)) continue;
    for (const specifier of moduleSpecifiers(readFileSync(absolute, 'utf8'), absolute)) {
      if (!specifier.startsWith('.')) continue;
      const target = toRepoPath(root, path.resolve(path.dirname(absolute), specifier.split(/[?#]/)[0]));
      if (!target.startsWith('..')) pending.push(target);
    }
  }
  return closure;
}

// Repository paths named by string values in a JSON file (evidence, receipts).
export function referencedPaths(file, root = REPO_ROOT) {
  const absolute = path.join(root, file);
  if (!existsSync(absolute)) throw new Error(`referencedBy file not found: ${file}`);
  const found = new Set([file]);
  (function walk(value) {
    if (typeof value === 'string') {
      const candidate = value.replace(/^\.\//, '').split('#')[0];
      if (/^[\w.-]+(?:\/[\w.@+-]+)+$/.test(candidate) && existsSync(path.join(root, candidate))) found.add(candidate);
    } else if (Array.isArray(value)) value.forEach(walk);
    else if (value && typeof value === 'object') Object.values(value).forEach(walk);
  })(JSON.parse(readFileSync(absolute, 'utf8')));
  return found;
}

// Exact inputs are read once per process; the checked files do not change mid-run.
const exactInputCache = new Map();
function exactInputs(name, entry, scripts, root) {
  const key = `${root}\0${name}\0${scripts[name]}\0${(entry.referencedBy ?? []).join('\0')}`;
  if (!exactInputCache.has(key)) {
    const exact = localImportClosure(scriptEntryFiles(name, scripts, root), root);
    for (const json of entry.referencedBy ?? []) for (const file of referencedPaths(json, root)) exact.add(file);
    exactInputCache.set(key, exact);
  }
  return exactInputCache.get(key);
}

export function selectTriggeredChecks(policy, scripts, changedFiles, root = REPO_ROOT) {
  const changed = [...changedFiles];
  if (changed.some((file) => GATE_FILES.includes(file))) {
    return Object.keys(policy.triggered).map((name) => ({ name, reason: 'check gate definition changed' }));
  }
  const selected = [];
  for (const [name, entry] of Object.entries(policy.triggered)) {
    const globs = entry.inputs.map(globToRegExp);
    const exact = exactInputs(name, entry, scripts, root);
    const hit = changed.find((file) => exact.has(file) || globs.some((glob) => glob.test(file)));
    if (hit) selected.push({ name, reason: hit });
  }
  return selected;
}

function git(args, root) {
  const result = spawnSync('git', args, { cwd: root, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`git ${args.join(' ')} failed: ${result.stderr.trim()}`);
  return result.stdout.split('\n').map((line) => line.trim()).filter(Boolean);
}

// Committed changes since the merge base, plus staged, unstaged and untracked files.
export function changedFilesSince(base, root = REPO_ROOT) {
  const [mergeBase] = git(['merge-base', base, 'HEAD'], root);
  if (!mergeBase) throw new Error(`No merge base between ${base} and HEAD`);
  return new Set([
    ...git(['diff', '--name-only', mergeBase, 'HEAD'], root),
    ...git(['diff', '--name-only', 'HEAD'], root),
    ...git(['ls-files', '--others', '--exclude-standard'], root),
  ]);
}

function parseArgs(argv) {
  const options = { behavior: false, changed: false, all: false, list: false, base: 'origin/main' };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--behavior') options.behavior = true;
    else if (arg === '--changed') options.changed = true;
    else if (arg === '--all') options.all = true;
    else if (arg === '--list') options.list = true;
    else if (arg === '--base') {
      options.base = argv[i + 1];
      if (!options.base || options.base.startsWith('--')) throw new Error('Missing value for --base');
      i += 1;
    } else throw new Error(`Unknown argument: ${arg}\n${USAGE}`);
  }
  if (!options.behavior && !options.changed && !options.all) throw new Error(USAGE);
  if (options.all && options.changed) throw new Error('--all already includes every triggered check; omit --changed');
  return options;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const { scripts } = JSON.parse(readFileSync(path.join(REPO_ROOT, 'package.json'), 'utf8'));
  const policy = JSON.parse(readFileSync(path.join(REPO_ROOT, POLICY_PATH), 'utf8'));
  const errors = validatePolicy(policy, scripts);
  if (errors.length) throw new Error(`${POLICY_PATH} is invalid:\n- ${errors.join('\n- ')}`);

  let triggered = [];
  if (options.all) triggered = Object.keys(policy.triggered).map((name) => ({ name, reason: '--all' }));
  else if (options.changed) triggered = selectTriggeredChecks(policy, scripts, changedFilesSince(options.base), REPO_ROOT);
  const behavior = options.behavior || options.all ? policy.behavior : [];
  const names = [...triggered.map((item) => item.name), ...behavior];

  if (options.list) {
    console.log(JSON.stringify({ base: options.changed ? options.base : null, behavior, triggered }, null, 2));
    return;
  }
  for (const item of triggered) console.log(`[check-gate] triggered ${item.name} (${item.reason})`);
  if (!names.length) {
    console.log('[check-gate] no checks selected');
    return;
  }
  const result = spawnSync(process.execPath, ['tools/run-node-tests.js', '--scripts', ...names], {
    cwd: REPO_ROOT,
    stdio: 'inherit',
  });
  if (result.error) throw result.error;
  process.exitCode = result.status ?? 1;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  }
}
