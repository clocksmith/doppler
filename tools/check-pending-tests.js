#!/usr/bin/env node
// Verify that every tests/**/*.pending.test.js file is registered in
// tools/policies/pending-tests-policy.json with an owner and either an
// expiry date or a tracking issue. Expired or orphaned entries fail the
// check so pending-feature debt stays visible and owned.

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const POLICY_PATH = path.join(ROOT, 'tools/policies/pending-tests-policy.json');
const TESTS_ROOT = path.join(ROOT, 'tests');

function walk(dir, acc = []) {
  if (!fs.existsSync(dir)) return acc;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.')) continue;
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(p, acc);
    else if (entry.name.endsWith('.pending.test.js')) acc.push(p);
  }
  return acc;
}

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function loadPolicy() {
  if (!fs.existsSync(POLICY_PATH)) {
    return { entries: [], errors: [`policy: file missing at ${path.relative(ROOT, POLICY_PATH)}`] };
  }
  let raw;
  try {
    raw = JSON.parse(fs.readFileSync(POLICY_PATH, 'utf8'));
  } catch (err) {
    return { entries: [], errors: [`policy: invalid JSON (${err.message})`] };
  }
  const entries = Array.isArray(raw?.entries) ? raw.entries : [];
  const errors = [];
  const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;
  entries.forEach((entry, idx) => {
    const tag = `policy[${idx}]`;
    for (const field of ['path', 'feature', 'owner', 'created']) {
      if (typeof entry[field] !== 'string' || entry[field].length === 0) {
        errors.push(`${tag}: missing required field "${field}"`);
      }
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
    if (!Array.isArray(entry.expectedMissing) || entry.expectedMissing.length === 0) {
      errors.push(`${tag}: expectedMissing must be a non-empty array`);
    }
  });
  return { entries, errors };
}

const policy = loadPolicy();
const pendingFiles = walk(TESTS_ROOT).map((p) => path.relative(ROOT, p));
const policyByPath = new Map();
for (const entry of policy.entries) {
  if (typeof entry.path === 'string') policyByPath.set(entry.path, entry);
}

const today = todayIso();
const unregistered = pendingFiles.filter((p) => !policyByPath.has(p));
const orphaned = [...policyByPath.keys()].filter((p) => !fs.existsSync(path.join(ROOT, p)));
const expired = policy.entries.filter((e) => e.expires && e.expires < today);

const hasFailure =
  policy.errors.length > 0 ||
  unregistered.length > 0 ||
  orphaned.length > 0 ||
  expired.length > 0;

if (!hasFailure) {
  console.log(
    `[pending-tests:check] ${pendingFiles.length} pending file(s), ${policy.entries.length} policy entr${policy.entries.length === 1 ? 'y' : 'ies'} — all owned, none expired`
  );
  process.exit(0);
}

if (policy.errors.length > 0) {
  console.error('[pending-tests:check] policy schema errors:');
  for (const msg of policy.errors) console.error(`  ${msg}`);
}

if (unregistered.length > 0) {
  console.error(`[pending-tests:check] ${unregistered.length} unregistered *.pending.test.js file(s) (add a policy entry):`);
  for (const p of unregistered) console.error(`  ${p}`);
}

if (orphaned.length > 0) {
  console.error(`[pending-tests:check] ${orphaned.length} orphaned policy entr${orphaned.length === 1 ? 'y' : 'ies'} (file missing — remove or restore):`);
  for (const p of orphaned) console.error(`  ${p}`);
}

if (expired.length > 0) {
  console.error(`[pending-tests:check] ${expired.length} expired policy entr${expired.length === 1 ? 'y' : 'ies'} (expiry < ${today}):`);
  for (const entry of expired) {
    console.error(`  ${entry.path} owner=${entry.owner} expired=${entry.expires}`);
  }
}

process.exit(1);
