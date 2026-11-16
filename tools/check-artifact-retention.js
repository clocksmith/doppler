#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ROOTS = ['artifacts', 'reports'];
const POLICY = 'tools/policies/artifact-retention.json';

export function summarizeArtifactInventory(inventory) {
  const campaigns = {};
  for (const item of [...inventory].sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0)) {
    const campaign = item.path.split('/').slice(0, 2).join('/');
    campaigns[campaign] ??= { files: 0, bytes: 0, records: [] };
    const entry = campaigns[campaign];
    entry.files++;
    entry.bytes += item.bytes;
    entry.records.push(`${item.path}\0${item.objectId}\0${item.bytes}\n`);
  }
  return Object.fromEntries(Object.entries(campaigns).map(([campaign, entry]) => [campaign, {
    files: entry.files,
    bytes: entry.bytes,
    inventoryHash: `sha256:${createHash('sha256').update(entry.records.join('')).digest('hex')}`,
  }]));
}

export function validateArtifactRetention(policy, inventory, today = new Date().toISOString().slice(0, 10)) {
  const errors = [];
  if (policy.schemaVersion !== 1) errors.push('artifact retention schemaVersion must be 1');
  const campaigns = summarizeArtifactInventory(inventory);
  for (const [name, retention] of Object.entries(policy.retentionClasses ?? {})) {
    if (!retention.owner || !retention.reason || !retention.removalCondition) {
      errors.push(`retention class ${name} requires owner, reason and removalCondition`);
    }
    if (retention.retainUntil !== undefined) {
      if (!/^\d{4}-\d{2}-\d{2}$/u.test(retention.retainUntil)
        || Number.isNaN(Date.parse(`${retention.retainUntil}T00:00:00Z`))
        || new Date(`${retention.retainUntil}T00:00:00Z`).toISOString().slice(0, 10) !== retention.retainUntil) {
        errors.push(`retention class ${name} has an invalid retainUntil date`);
      } else if (retention.retainUntil < today) {
        errors.push(`retention class ${name} expired on ${retention.retainUntil}`);
      }
    }
  }
  for (const [campaign, actual] of Object.entries(campaigns)) {
    const record = policy.campaigns?.[campaign];
    if (!record) {
      errors.push(`tracked receipts have no retention record: ${campaign}`);
      continue;
    }
    if (!Object.hasOwn(policy.retentionClasses ?? {}, record.retentionClass)) {
      errors.push(`${campaign}: unknown retention class ${record.retentionClass}`);
    }
    if (record.files !== actual.files || record.bytes !== actual.bytes || record.inventoryHash !== actual.inventoryHash) {
      errors.push(`${campaign}: receipt inventory changed; review retention and update its exact inventory`);
    }
  }
  for (const campaign of Object.keys(policy.campaigns ?? {})) {
    if (!Object.hasOwn(campaigns, campaign)) errors.push(`stale retention record: ${campaign}`);
  }
  return errors;
}

export function readTrackedArtifactInventory(root = ROOT) {
  const git = (args, options = {}) => execFileSync('git', args, {
    cwd: root, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024, ...options,
  });
  const tracked = git(['ls-files', '--stage', '-z', '--', ...ROOTS]).split('\0').filter(Boolean).map(record => {
    const [header, file] = [record.slice(0, record.indexOf('\t')), record.slice(record.indexOf('\t') + 1)];
    const [, objectId, stage] = header.split(' ');
    if (stage !== '0') throw new Error(`Unmerged receipt: ${file}`);
    return { path: file, objectId };
  });
  const changed = new Set(git(['diff', '--name-only', '-z', '--', ...ROOTS]).split('\0').filter(Boolean));
  for (const item of tracked) {
    if (!changed.has(item.path)) continue;
    if (!fs.existsSync(path.join(root, item.path))) throw new Error(`Tracked receipt is missing: ${item.path}`);
    item.objectId = git(['hash-object', `--path=${item.path}`, '--', item.path]).trim();
    item.bytes = fs.statSync(path.join(root, item.path)).size;
  }
  const ids = [...new Set(tracked.filter(item => item.bytes === undefined).map(item => item.objectId))];
  if (ids.length === 0) return tracked;
  const rows = git(['cat-file', '--batch-check=%(objectname) %(objectsize)'], { input: ids.join('\n') + '\n' });
  const sizes = new Map(rows.trim().split('\n').map(row => {
    const [id, size] = row.split(' ');
    if (!Number.isSafeInteger(Number(size))) throw new Error(`Cannot inspect retained blob: ${id}`);
    return [id, Number(size)];
  }));
  return tracked.map(item => ({ ...item, bytes: item.bytes ?? sizes.get(item.objectId) }));
}

function main() {
  const inventory = readTrackedArtifactInventory();
  if (process.argv[2] === '--inventory') {
    const prefix = process.argv[3] ?? '';
    console.log(JSON.stringify(summarizeArtifactInventory(inventory.filter(item => item.path.startsWith(prefix))), null, 2));
    return;
  }
  if (process.argv.length > 2) throw new Error('Usage: node tools/check-artifact-retention.js [--inventory [prefix]]');
  const policy = JSON.parse(fs.readFileSync(path.join(ROOT, POLICY), 'utf8'));
  const errors = validateArtifactRetention(policy, inventory);
  if (errors.length) throw new Error(errors.join('\n'));
  console.log(`artifact retention: ${inventory.length} tracked receipts in ${Object.keys(policy.campaigns).length} exact campaign inventories`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
