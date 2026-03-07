#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = process.cwd();
const agentsPath = path.join(root, 'AGENTS.md');

const strictMode = process.argv.includes('--strict');
const maxAgeIndex = process.argv.findIndex((arg) => arg === '--max-age-days' || arg === '--maxAgeDays');
const maxAgeDays = Number.parseInt(
  maxAgeIndex >= 0 ? (process.argv[maxAgeIndex + 1] || '0') : '0',
  10
);
const effectiveMaxAge = Number.isFinite(maxAgeDays) && maxAgeDays > 0 ? maxAgeDays : 0;

export const REQUIRED_SKILL_SYMLINKS = Object.freeze([
  ['.claude/skills', '../skills'],
  ['.gemini/skills', '../skills'],
  ['.codex/skills', '../skills'],
]);

function rel(p) {
  return path.relative(root, p);
}

function readFileSafe(file) {
  try {
    return fs.readFileSync(file, 'utf8');
  } catch (error) {
    throw new Error(`failed to read ${file}: ${error.message}`);
  }
}

function checkExists(target, reason, issues) {
  const absolute = path.resolve(root, target);
  try {
    const stat = fs.lstatSync(absolute);
    if (!stat.isSymbolicLink()) {
      const ageMs = Date.now() - stat.mtimeMs;
      const ageDays = Math.floor(ageMs / (1000 * 60 * 60 * 24));
      if (effectiveMaxAge > 0 && ageDays > effectiveMaxAge) {
        issues.warnings.push({
          path: target,
          reason: `${reason} (stale by ${ageDays}d > ${effectiveMaxAge}d)`,
          source: 'AGE',
        });
      }
    }
    return true;
  } catch {
    issues.errors.push({
      path: target,
      reason: reason,
      source: 'MISSING',
    });
    return false;
  }
}

function checkSymlink(file, target, issues) {
  try {
    const stat = fs.lstatSync(path.resolve(root, file));
    if (!stat.isSymbolicLink()) {
      issues.errors.push({
        path: file,
        reason: `expected symlink to ${target}`,
        source: 'SYMLINK',
      });
      return;
    }

    const actual = fs.readlinkSync(path.resolve(root, file));
    if (actual !== target) {
      issues.errors.push({
        path: file,
        reason: `symlink points to ${actual}, expected ${target}`,
        source: 'SYMLINK',
      });
    }
  } catch (error) {
    issues.errors.push({
      path: file,
      reason: `symlink missing or unreadable (${error.message})`,
      source: 'SYMLINK',
    });
  }
}

function normalizeMarkdownPath(raw) {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) return null;
  if (trimmed === '#') return null;
  if (trimmed.includes(' ')) return null;
  if (trimmed.startsWith('node ') || trimmed.startsWith('npm ')) return null;
  if (trimmed.startsWith('../')) return null;

  const withoutAnchor = trimmed.split('#')[0];
  const withoutWhitespace = withoutAnchor.replace(/\s+/g, ' ').trim();
  if (!withoutWhitespace) return null;
  if (/<[^>]+>/.test(withoutWhitespace)) return null;

  const withoutGlob = withoutWhitespace.replace(/\\*.*$/, '');
  if (!withoutGlob) return null;

  return withoutGlob.replace(/^\.\/+/, '');
}

function looksLikePath(token) {
  if (!token || token.includes(':') || token.startsWith('$')) return false;
  if (!token.includes('/')) return false;
  if (token.includes('<') || token.includes('>')) return false;
  return true;
}

function collectReferencedPaths(agentsText) {
  const refs = new Set();

  const inlineLinkRegex = /\[[^\]]+\]\(([^)\s]+)\)/g;
  let match;
  while ((match = inlineLinkRegex.exec(agentsText)) !== null) {
    const p = normalizeMarkdownPath(match[1]);
    if (p && looksLikePath(p)) {
      refs.add(p);
    }
  }

  const codeBacktickRegex = /`([^`\n]+)`/g;
  while ((match = codeBacktickRegex.exec(agentsText)) !== null) {
    const token = normalizeMarkdownPath(match[1]);
    if (!token) continue;
    if (!looksLikePath(token)) continue;
    refs.add(token);
  }

  const bullets = /\s*-\s*`([^`]+)`\s*:\s*`([^`]+)`/g;
  while ((match = bullets.exec(agentsText)) !== null) {
    if (looksLikePath(match[1])) refs.add(match[1]);
    if (looksLikePath(match[2])) refs.add(match[2]);
  }

  return [...refs];
}

function ensureSkillFiles(issues) {
  const skillPaths = [
    'skills/doppler-debug/SKILL.md',
    'skills/doppler-bench/SKILL.md',
    'skills/doppler-perf-squeeze/SKILL.md',
    'skills/doppler-convert/SKILL.md',
    'skills/doppler-kernel-reviewer/SKILL.md',
  ];
  for (const skill of skillPaths) {
    checkExists(skill, 'required skill file missing', issues);
  }
}

function ensureScripts(issues) {
  let pkg;
  try {
    pkg = JSON.parse(readFileSafe(path.join(root, 'package.json')));
  } catch (error) {
    issues.errors.push({
      path: 'package.json',
      reason: `failed to parse package.json: ${error.message}`,
      source: 'PACKAGE',
    });
    return;
  }
  const scripts = pkg.scripts || {};
  const required = ['agents:verify', 'convert', 'debug', 'bench', 'verify:model'];
  for (const key of required) {
    if (!scripts[key]) {
      issues.warnings.push({
        path: `package.json scripts.${key}`,
        reason: `missing script ${key}`,
        source: 'SCRIPT',
      });
    }
  }
}

function main() {
  const agents = readFileSafe(agentsPath);
  const refs = collectReferencedPaths(agents);
  const issues = { errors: [], warnings: [] };

  const knownPaths = [
    'docs/architecture.md',
    'docs/rdrr-format.md',
    'docs/config.md',
    'src/inference/pipelines/text.js',
    'src/tooling/command-api.js',
    'tools/doppler-cli.js',
    'tools/policies/agent-parity-policy.json',
    'benchmarks/vendors/benchmark-policy.json',
    'tools/verify-agent-parity.js',
    'benchmarks/vendors/registry.json',
    'benchmarks/vendors/workloads.json',
    'benchmarks/vendors/capabilities.json',
    'benchmarks/vendors/harnesses',
    'benchmarks/vendors/results',
    'benchmarks/vendors/README.md',
    'skills/README.md',
    'src/tooling/browser-command-runner.js',
    'src/tooling/node-command-runner.js',
    'tools/vendor-bench.js',
  ];

  for (const p of [...refs, ...knownPaths]) {
    if (!p || p.startsWith('AGENTS.md')) {
      continue;
    }
    checkExists(p, `AGENTS.md references missing path: ${p}`, issues);
  }

  checkSymlink('CLAUDE.md', 'AGENTS.md', issues);
  checkSymlink('GEMINI.md', 'AGENTS.md', issues);
  for (const [linkPath, target] of REQUIRED_SKILL_SYMLINKS) {
    checkSymlink(linkPath, target, issues);
  }

  ensureSkillFiles(issues);
  ensureScripts(issues);

  const summary = {
    checked: refs.length + knownPaths.length,
    missing: issues.errors.length,
    stale: issues.warnings.filter((item) => item.source === 'AGE').length,
    warning: issues.warnings.length,
    strict: strictMode,
  };

  const line = (kind, item) => `  - ${kind}: ${item.path} (${item.reason})`;
  if (issues.errors.length > 0) {
    console.error('[agents-freshness] errors');
    for (const item of issues.errors) {
      console.error(line('error', item));
    }
  }
  if (issues.warnings.length > 0) {
    if (effectiveMaxAge > 0) {
      console.warn('[agents-freshness] warnings');
    } else {
      console.warn('[agents-freshness] warnings (add --max-age-days to enable freshness checks)');
    }
    for (const item of issues.warnings) {
      console.warn(line('warning', item));
    }
  }
  if (issues.errors.length === 0 && issues.warnings.length === 0) {
    console.log('[agents-freshness] OK: AGENTS references and symlink/skill/script contract are valid.');
  } else {
    console.log(`[agents-freshness] summary: checked=${summary.checked} missing=${summary.missing} stale=${summary.stale} warnings=${summary.warning}`);
  }

  if (issues.errors.length > 0) process.exitCode = 1;
  else if (strictMode && issues.warnings.length > 0) process.exitCode = 1;
  else process.exitCode = 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
