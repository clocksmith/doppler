#!/usr/bin/env node
// Fails when a Doppler-owned enum vocabulary gains a snake_case or camelCase
// value. Doppler-owned enum values are kebab-case (lowercase letters, digits
// and hyphens). Values that mirror an external format (GGUF/SafeTensors keys,
// Hugging Face config fields, WGSL/WebGPU identifiers, dtype names) are listed
// in tools/policies/kebab-enum-values.json; there is no allowlist for values
// that are merely published: those are renamed and read through
// src/config/legacy-enum-values.js.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  LEGACY_ENUM_VALUE_ALIASES,
  LEGACY_ENUM_VALUE_FIELDS,
  LEGACY_OUTCOME_VALUE_FIELDS,
} from '../src/config/legacy-enum-values.js';

const SNAKE = /^[a-z][a-z0-9]*(_[a-z0-9]+)+$/;
const CAMEL = /^[a-z]+[A-Z][A-Za-z0-9]*$/;
const ENUM_ARRAY_NAME = /(MODES|KINDS|TYPES|VALUES|STATUSES|STATES|CLASSES|POLICIES|SURFACES|INTENTS|LAYOUTS|STAGES|LANES|REASONS|_SET)$/;
const STRING_ARRAY = /(?:const|let)\s+([A-Z][A-Z0-9_]*)\s*=\s*(?:Object\.freeze\(\s*)?(?:new Set\(\s*)?\[([^\]]*)\]/g;

function isNonKebab(value) {
  return SNAKE.test(value) || CAMEL.test(value);
}

function listFiles(root, scopes) {
  const out = execFileSync('git', ['-C', root, 'ls-files', '--', ...scopes], { maxBuffer: 1 << 28 }).toString();
  return out.split('\n').filter(Boolean);
}

function matchesAny(file, patterns) {
  return patterns.some((pattern) => new RegExp(pattern).test(file));
}

function walkJson(value, key, visit) {
  if (typeof value === 'string') {
    visit(key, value);
  } else if (Array.isArray(value)) {
    for (const entry of value) walkJson(entry, key, visit);
  } else if (value && typeof value === 'object') {
    for (const [childKey, entry] of Object.entries(value)) walkJson(entry, childKey, visit);
  }
}

export function checkKebabEnumValues({ root, policy }) {
  const violations = [];
  const externalValues = new Set(Object.values(policy.externalValues).flat());
  const legacyLiteralFiles = policy.legacyLiteralAllowedFiles ?? [];
  const enumFields = new Set([
    ...[...LEGACY_ENUM_VALUE_FIELDS].filter((field) => !LEGACY_OUTCOME_VALUE_FIELDS.has(field)),
    ...(policy.extraEnumFields ?? []),
  ]);
  const legacy = LEGACY_ENUM_VALUE_ALIASES;

  for (const file of listFiles(root, policy.jsonScopes)) {
    if (!file.endsWith('.json') || matchesAny(file, policy.excludeFiles)) continue;
    const absolute = path.join(root, file);
    if (!fs.existsSync(absolute)) continue;
    let parsed;
    try {
      parsed = JSON.parse(fs.readFileSync(absolute, 'utf8'));
    } catch {
      continue;
    }
    walkJson(parsed, null, (key, value) => {
      if (Object.prototype.hasOwnProperty.call(legacy, value)) {
        // Legacy spellings are allowed only in files that must keep them (read-time aliases, digest-bound inputs).
        if (!matchesAny(file, legacyLiteralFiles)) {
          violations.push(`${file}: legacy value "${value}" (use "${legacy[value]}")`);
        }
      } else if (key && enumFields.has(key) && isNonKebab(value) && !externalValues.has(value)) {
        violations.push(`${file}: ${key}="${value}" is not kebab-case`);
      }
    });
  }

  for (const file of listFiles(root, policy.jsScopes)) {
    if (!file.endsWith('.js') || matchesAny(file, policy.excludeFiles)) continue;
    const absolute = path.join(root, file);
    if (!fs.existsSync(absolute)) continue;
    const source = fs.readFileSync(absolute, 'utf8');
    for (const match of source.matchAll(/(['"`])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\1/g)) {
      if (Object.prototype.hasOwnProperty.call(legacy, match[2]) && !matchesAny(file, legacyLiteralFiles)) {
        violations.push(`${file}: legacy value "${match[2]}" (use "${legacy[match[2]]}")`);
      }
    }
    for (const match of source.matchAll(STRING_ARRAY)) {
      if (!ENUM_ARRAY_NAME.test(match[1])) continue;
      const members = [...match[2].matchAll(/(['"])([^'"\n]*)\1/g)].map((entry) => entry[2]);
      for (const member of members) {
        if (isNonKebab(member) && !externalValues.has(member)) {
          violations.push(`${file}: ${match[1]} member "${member}" is not kebab-case`);
        }
      }
    }
  }
  return [...new Set(violations)].sort();
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const policyPath = process.argv.includes('--policy')
    ? path.resolve(process.argv[process.argv.indexOf('--policy') + 1])
    : path.join(root, 'tools/policies/kebab-enum-values.json');
  const policy = JSON.parse(fs.readFileSync(policyPath, 'utf8'));
  const checkRoot = process.argv.includes('--root')
    ? path.resolve(process.argv[process.argv.indexOf('--root') + 1])
    : root;
  const violations = checkKebabEnumValues({ root: checkRoot, policy });
  if (violations.length > 0) {
    console.error(`[kebab-enum-values] ${violations.length} non-kebab enum value(s):`);
    for (const violation of violations) console.error(`  - ${violation}`);
    process.exit(1);
  }
  console.log('[kebab-enum-values] Doppler-owned enum values are kebab-case');
}
