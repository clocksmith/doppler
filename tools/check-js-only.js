#!/usr/bin/env node
// Doppler is JavaScript-only (WGSL shaders are fine). This check fails on any
// tracked Python or Go source or project file, and on any script, workflow or
// package script that runs a Python or Go toolchain.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// The checker and its fixture test necessarily spell out the patterns they reject.
const SELF = new Set(['tools/check-js-only.js', 'tests/tooling/js-only-check.test.js']);

const FORBIDDEN_EXTENSIONS = new Set(['.py', '.pyi', '.pyx', '.pxd', '.ipynb', '.go']);
const FORBIDDEN_BASENAMES = new Set([
  'go.mod', 'go.sum', 'go.work', 'pyproject.toml', 'setup.py', 'setup.cfg', 'Pipfile', 'Pipfile.lock',
  'poetry.lock', 'uv.lock', 'tox.ini', 'conftest.py',
]);
const REQUIREMENTS = /^requirements(?:[-.][\w.-]+)?\.txt$/u;

const SCANNED_EXTENSIONS = new Set(['.js', '.cjs', '.mjs', '.ts', '.cts', '.mts', '.sh', '.bash', '.zsh']);
const INTERPRETERS = 'python[0-9.]*|pip[0-9.]*|pytest|uv|go';
const BOUNDARY = String.raw`(?:^|[\s;&|(\x60'"=])`;
const RULES = Object.freeze([
  {
    id: 'spawns-interpreter',
    pattern: new RegExp(String.raw`\b(?:spawn|spawnSync|exec|execSync|execFile|execFileSync|execFileAsync|execa|fork)\s*\(\s*['"\x60](?:${INTERPRETERS})['"\x60\s]`, 'u'),
  },
  {
    id: 'python-command',
    pattern: new RegExp(String.raw`${BOUNDARY}(?:python[0-9.]*|pytest|pip[0-9.]*)(?:\s+-[A-Za-z]|\s+-(?:\s|$)|\s+[^\s'"\x60]*\.py\b|\s+install\b)`, 'mu'),
  },
  { id: 'go-command', pattern: new RegExp(String.raw`${BOUNDARY}go\s+(?:run|build|test|install|generate|vet|mod|get)\b`, 'mu') },
  { id: 'uv-command', pattern: new RegExp(String.raw`${BOUNDARY}uv\s+(?:run|pip|sync|venv)\b`, 'mu') },
  { id: 'python-environment', pattern: /\b[A-Z0-9_]*PYTHON[A-Z0-9_]*\b|\/bin\/python|\bpython(?:Path|Bin|Executable|Interpreter|Command)\b|[a-z]Python\b/u },
  {
    // A Python interpreter passed as a command with an argument list, or
    // selected as a command value: run('python3', [...]), env.X || 'python3'.
    id: 'interpreter-argument',
    pattern: /['"\x60](?:python[0-9.]*|pytest|pip[0-9.]*)['"\x60]\s*,\s*\[|(?:=|\|\||\?\?|:)\s*['"\x60](?:python[0-9.]*|pytest)['"\x60]/u,
  },
  { id: 'toolchain-setup', pattern: /actions\/setup-(?:python|go)@|\bsetup-python\b|\bsetup-go\b/u },
]);

function isScanned(file) {
  if (file.startsWith('.github/workflows/') && /\.ya?ml$/u.test(file)) return true;
  // A package.json copied into a results directory is a recorded source snapshot, not a runnable package.
  if (path.posix.basename(file) === 'package.json') return !file.split('/').includes('results');
  return SCANNED_EXTENSIONS.has(path.posix.extname(file));
}

function packageScripts(text) {
  try {
    const scripts = JSON.parse(text).scripts;
    return scripts && typeof scripts === 'object' ? Object.values(scripts).join('\n') : '';
  } catch {
    return '';
  }
}

export function findJsOnlyViolations(files) {
  const violations = [];
  for (const { path: file, text } of files) {
    if (SELF.has(file)) continue;
    const base = path.posix.basename(file);
    if (FORBIDDEN_EXTENSIONS.has(path.posix.extname(file)) || FORBIDDEN_BASENAMES.has(base) || REQUIREMENTS.test(base)) {
      violations.push({ path: file, rule: 'forbidden-language-file' });
      continue;
    }
    if (typeof text !== 'string') continue;
    const firstLine = text.slice(0, text.indexOf('\n') === -1 ? text.length : text.indexOf('\n'));
    if (/^#!.*\b(?:python[0-9.]*|go\s+run)\b/u.test(firstLine)) {
      violations.push({ path: file, rule: 'interpreter-shebang' });
      continue;
    }
    if (!isScanned(file)) continue;
    const body = base === 'package.json' ? packageScripts(text) : text;
    const lines = body.split('\n');
    for (const rule of RULES) {
      const index = lines.findIndex(line => rule.pattern.test(line));
      if (index !== -1) {
        violations.push({ path: file, rule: rule.id, line: base === 'package.json' ? undefined : index + 1, text: lines[index].trim().slice(0, 160) });
      }
    }
  }
  return violations;
}

export function readTrackedFiles(root = ROOT) {
  const listed = execFileSync('git', ['ls-files', '-z'], { cwd: root, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 })
    .split('\0').filter(Boolean);
  return listed.map(file => {
    const absolute = path.join(root, file);
    let text = null;
    if (!FORBIDDEN_EXTENSIONS.has(path.posix.extname(file))) {
      try {
        const stat = fs.lstatSync(absolute);
        if (stat.isFile() && stat.size <= 4 * 1024 * 1024) text = fs.readFileSync(absolute, 'utf8');
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
        return null;
      }
    }
    return { path: file, text };
  }).filter(Boolean);
}

function main() {
  const violations = findJsOnlyViolations(readTrackedFiles());
  if (process.argv.includes('--json')) {
    console.log(JSON.stringify(violations, null, 2));
  } else {
    for (const v of violations) console.error(`${v.path}${v.line ? `:${v.line}` : ''}: ${v.rule}${v.text ? `: ${v.text}` : ''}`);
  }
  if (violations.length) {
    console.error(`js-only: ${violations.length} violation(s). Doppler code is JavaScript only (WGSL shaders allowed); no Python or Go.`);
    process.exitCode = 1;
    return;
  }
  console.log('js-only: no Python or Go files and no Python or Go invocations in tracked scripts');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
