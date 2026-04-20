#!/usr/bin/env node
// Verify that every `export` value in a .js file has a matching declaration in
// the sibling .d.ts (and vice-versa, modulo type-only exports). Catches the
// "runtime exports a symbol that types don't advertise" drift class.
//
// Scope: every src/**/*.js that has a sibling .d.ts and no `export * from`
// (wildcard re-exports can't be parity-checked by regex).
//
// Known-debt quarantine lives in tools/policies/exports-parity-allowlist.json.
// Entries must carry an owner and either an expiry date or a tracking issue.
// Expired or unused allowlist entries fail the check so debt cannot rot
// invisibly.

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const ALLOWLIST_PATH = path.join(ROOT, 'tools/policies/exports-parity-allowlist.json');

function walk(dir, acc = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(p, acc);
    else if (entry.name.endsWith('.js')) acc.push(p);
  }
  return acc;
}

// Strip `// line` and `/* block */` comments from a grouped-export body
// so names after an inline comment are still extracted. Previously any
// piece of the form "// foo\n  someName" fell through the identifier
// regex and the export was silently missed on one side.
function stripComments(s) {
  return s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

function jsExports(src) {
  const names = new Set();
  const patterns = [
    // `function` or `function*` (generator), optionally `async`.
    /^export\s+(?:async\s+)?function\s*\*?\s*([A-Za-z_$][A-Za-z0-9_$]*)/gm,
    /^export\s+(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)/gm,
    /^export\s+(?:async\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)/gm,
  ];
  for (const re of patterns) for (const m of src.matchAll(re)) names.add(m[1]);
  for (const m of src.matchAll(/^export\s*\{([^}]+)\}/gm)) {
    for (const piece of stripComments(m[1]).split(',')) {
      const parts = piece.trim().split(/\s+as\s+/);
      const exported = (parts[1] || parts[0]).trim();
      if (exported && /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(exported)) names.add(exported);
    }
  }
  return names;
}

function dtsValueExports(src) {
  const names = new Set();
  const patterns = [
    /^export\s+declare\s+(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)/gm,
    /^export\s+(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)/gm,
    /^export\s+declare\s+(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)/gm,
    /^export\s+(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)/gm,
    /^export\s+declare\s+(?:abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)/gm,
    /^export\s+(?:abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)/gm,
  ];
  for (const re of patterns) for (const m of src.matchAll(re)) names.add(m[1]);
  for (const m of src.matchAll(/^export\s*\{([^}]+)\}/gm)) {
    for (const piece of stripComments(m[1]).split(',')) {
      const cleaned = piece.replace(/^\s*type\s+/, '').trim();
      if (cleaned !== piece.trim()) continue;
      const parts = cleaned.split(/\s+as\s+/);
      const exported = (parts[1] || parts[0]).trim();
      if (exported && /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(exported)) names.add(exported);
    }
  }
  return names;
}

function hasWildcardExport(src) {
  return /^export\s*\*\s*from/m.test(src);
}

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function loadAllowlist() {
  if (!fs.existsSync(ALLOWLIST_PATH)) return { entries: [], errors: [] };
  let raw;
  try {
    raw = JSON.parse(fs.readFileSync(ALLOWLIST_PATH, 'utf8'));
  } catch (err) {
    return { entries: [], errors: [`allowlist: invalid JSON (${err.message})`] };
  }
  const entries = Array.isArray(raw?.entries) ? raw.entries : [];
  const errors = [];
  const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;
  const KIND = new Set(['onlyInJs', 'onlyInDts']);
  entries.forEach((entry, idx) => {
    const tag = `allowlist[${idx}]`;
    for (const field of ['path', 'kind', 'symbol', 'reason', 'owner', 'created']) {
      if (typeof entry[field] !== 'string' || entry[field].length === 0) {
        errors.push(`${tag}: missing required field "${field}"`);
      }
    }
    if (entry.kind && !KIND.has(entry.kind)) {
      errors.push(`${tag}: kind must be "onlyInJs" or "onlyInDts" (got "${entry.kind}")`);
    }
    if (entry.created && !ISO_DATE.test(entry.created)) {
      errors.push(`${tag}: created must be YYYY-MM-DD (got "${entry.created}")`);
    }
    const hasExpires = typeof entry.expires === 'string' && entry.expires.length > 0;
    const hasIssue = typeof entry.issue === 'string' && entry.issue.length > 0;
    if (!hasExpires && !hasIssue) {
      errors.push(`${tag}: must set either "expires" (YYYY-MM-DD) or "issue" (tracking URL)`);
    }
    if (hasExpires && !ISO_DATE.test(entry.expires)) {
      errors.push(`${tag}: expires must be YYYY-MM-DD (got "${entry.expires}")`);
    }
  });
  return { entries, errors };
}

function entryKey(file, kind, symbol) {
  return `${file}\u0000${kind}\u0000${symbol}`;
}

const allowlist = loadAllowlist();
const allowlistUsed = new Set();
const allowlistByKey = new Map();
for (const entry of allowlist.entries) {
  if (!entry.path || !entry.kind || !entry.symbol) continue;
  allowlistByKey.set(entryKey(entry.path, entry.kind, entry.symbol), entry);
}

const jsFiles = walk(path.join(ROOT, 'src'));
const drift = [];

for (const jsFile of jsFiles) {
  const dtsFile = jsFile.replace(/\.js$/, '.d.ts');
  if (!fs.existsSync(dtsFile)) continue;
  const jsSrc = fs.readFileSync(jsFile, 'utf8');
  const dtsSrc = fs.readFileSync(dtsFile, 'utf8');
  if (hasWildcardExport(jsSrc) || hasWildcardExport(dtsSrc)) continue;

  const jsE = jsExports(jsSrc);
  const dtsE = dtsValueExports(dtsSrc);
  const rel = path.relative(ROOT, jsFile);
  const raw = {
    onlyInJs: [...jsE].filter((n) => !dtsE.has(n)),
    onlyInDts: [...dtsE].filter((n) => !jsE.has(n)),
  };

  const unallowed = { onlyInJs: [], onlyInDts: [] };
  for (const kind of ['onlyInJs', 'onlyInDts']) {
    for (const symbol of raw[kind]) {
      const key = entryKey(rel, kind, symbol);
      if (allowlistByKey.has(key)) {
        allowlistUsed.add(key);
        continue;
      }
      unallowed[kind].push(symbol);
    }
  }
  if (unallowed.onlyInJs.length || unallowed.onlyInDts.length) {
    drift.push({ file: rel, onlyInJs: unallowed.onlyInJs, onlyInDts: unallowed.onlyInDts });
  }
}

const today = todayIso();
const expired = [];
const unused = [];
for (const entry of allowlist.entries) {
  if (!entry.path || !entry.kind || !entry.symbol) continue;
  const key = entryKey(entry.path, entry.kind, entry.symbol);
  if (entry.expires && entry.expires < today) {
    expired.push(entry);
    continue;
  }
  if (!allowlistUsed.has(key)) unused.push(entry);
}

const hasFailure =
  drift.length > 0 ||
  allowlist.errors.length > 0 ||
  expired.length > 0 ||
  unused.length > 0;

if (!hasFailure) {
  console.log('[exports-parity:check] all .js/.d.ts pairs agree on named exports');
  if (allowlist.entries.length > 0) {
    console.log(`[exports-parity:check] ${allowlist.entries.length} allowlist entr${allowlist.entries.length === 1 ? 'y' : 'ies'} active`);
  }
  process.exit(0);
}

if (allowlist.errors.length > 0) {
  console.error('[exports-parity:check] allowlist schema errors:');
  for (const msg of allowlist.errors) console.error(`  ${msg}`);
}

if (drift.length > 0) {
  console.error(`[exports-parity:check] ${drift.length} file(s) with JS/d.ts export drift:`);
  for (const item of drift) {
    console.error(`  ${item.file}`);
    if (item.onlyInJs.length) console.error(`    only in .js : ${item.onlyInJs.join(', ')}`);
    if (item.onlyInDts.length) console.error(`    only in .dts: ${item.onlyInDts.join(', ')}`);
  }
}

if (expired.length > 0) {
  console.error(`[exports-parity:check] ${expired.length} expired allowlist entr${expired.length === 1 ? 'y' : 'ies'} (expiry < ${today}):`);
  for (const entry of expired) {
    console.error(`  ${entry.path} [${entry.kind} ${entry.symbol}] owner=${entry.owner} expired=${entry.expires}`);
  }
}

if (unused.length > 0) {
  console.error(`[exports-parity:check] ${unused.length} unused allowlist entr${unused.length === 1 ? 'y' : 'ies'} (no matching drift — remove them):`);
  for (const entry of unused) {
    console.error(`  ${entry.path} [${entry.kind} ${entry.symbol}] owner=${entry.owner}`);
  }
}

process.exit(1);
