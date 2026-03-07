#!/usr/bin/env node
// tools/review-audit.js
// Append-only event log + materialized latest-state tracker for the src/ review.
//
// Subcommands:
//   seed    Walk src/ and write initial latest.jsonl rows (status: unreviewed)
//   event   Append one event to events.jsonl, then regen latest.jsonl
//   regen   Regenerate latest.jsonl from events.jsonl
//
// Files:
//   reports/review/src-audit/events.jsonl  -- append-only, one line per action
//   reports/review/src-audit/latest.jsonl  -- one row per file, rewritten on regen

import { readFileSync, writeFileSync, appendFileSync, existsSync, mkdirSync } from 'fs';
import { readdir } from 'fs/promises';
import { join, relative } from 'path';

const ROOT = new URL('..', import.meta.url).pathname;
const AUDIT_DIR = join(ROOT, 'reports/review/src-audit');
const EVENTS_FILE = join(AUDIT_DIR, 'events.jsonl');
const LATEST_FILE = join(AUDIT_DIR, 'latest.jsonl');

// Ownership rules — first match wins.
// Owner A: config/manifest/policy plane
// Owner B: runtime/kernel/inference plane
// Owner C: command/harness/surface plane
const OWNER_RULES = [
  // Exact files in src/inference/ that belong to A (manifest-first enforcement)
  { owner: 'A', agent: 'config-owner', test: f => f === 'src/inference/pipelines/text/execution-v0.js' },
  { owner: 'A', agent: 'config-owner', test: f => f === 'src/inference/pipelines/text/model-load.js' },
  // Exact file in src/inference/ that belongs to C (harness surface)
  { owner: 'C', agent: 'surface-owner', test: f => f === 'src/inference/browser-harness.js' },
  // A: config, rules, converter
  { owner: 'A', agent: 'config-owner', test: f =>
    f.startsWith('src/config/') || f.startsWith('src/rules/') || f.startsWith('src/converter/') },
  // B: inference (remainder), gpu, loader, memory, formats, training, generation, diffusion, energy
  { owner: 'B', agent: 'runtime-owner', test: f =>
    f.startsWith('src/inference/') || f.startsWith('src/gpu/') || f.startsWith('src/loader/') ||
    f.startsWith('src/memory/') || f.startsWith('src/formats/') || f.startsWith('src/training/') ||
    f.startsWith('src/generation/') || f.startsWith('src/diffusion/') || f.startsWith('src/energy/') },
  // C: everything else under src/
  { owner: 'C', agent: 'surface-owner', test: f => f.startsWith('src/') },
];

function assignOwner(relPath) {
  for (const rule of OWNER_RULES) {
    if (rule.test(relPath)) return { owner: rule.owner, agent: rule.agent };
  }
  return { owner: 'C', agent: 'surface-owner' };
}

async function walkSrc(dir) {
  const files = [];
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await walkSrc(full));
    } else if (entry.isFile() && /\.(js|wgsl)$/.test(entry.name) && !entry.name.endsWith('.d.js')) {
      files.push(full);
    }
  }
  return files;
}

function readJsonl(file) {
  if (!existsSync(file)) return [];
  return readFileSync(file, 'utf8')
    .split('\n')
    .filter(l => l.trim())
    .map(l => JSON.parse(l));
}

function writeJsonl(file, rows) {
  const content = rows.map(r => JSON.stringify(r)).join('\n');
  writeFileSync(file, content ? content + '\n' : '');
}

function ensureDir(dir) {
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

function nextSeq() {
  return readJsonl(EVENTS_FILE).length + 1;
}

// ── seed ──────────────────────────────────────────────────────────────────────

async function seed() {
  ensureDir(AUDIT_DIR);
  const srcDir = join(ROOT, 'src');
  const allFiles = (await walkSrc(srcDir)).sort();

  const existing = new Set(readJsonl(LATEST_FILE).map(r => r.path));
  const newRows = [];

  for (const abs of allFiles) {
    const rel = relative(ROOT, abs);
    if (existing.has(rel)) continue;
    const { owner, agent } = assignOwner(rel);
    newRows.push({
      path: rel,
      owner,
      last_agent: agent,
      review_status: 'unreviewed',
      last_seq: null,
      passes: [],
      open_findings: [],
      closed_findings: [],
      last_reviewed_at: null,
    });
  }

  if (newRows.length === 0) {
    console.log('All src/ files already present in latest.jsonl.');
    return;
  }

  const all = [...readJsonl(LATEST_FILE), ...newRows]
    .sort((a, b) => a.path.localeCompare(b.path));
  writeJsonl(LATEST_FILE, all);
  console.log(`Seeded ${newRows.length} file(s) → ${LATEST_FILE}`);
}

// ── regen ─────────────────────────────────────────────────────────────────────

function regenLatest() {
  ensureDir(AUDIT_DIR);
  const events = readJsonl(EVENTS_FILE);

  // Start from whatever is already in latest (preserves seeded rows with no events yet)
  const byPath = new Map(readJsonl(LATEST_FILE).map(r => [r.path, { ...r }]));

  for (const ev of events) {
    const row = byPath.get(ev.path) ?? {
      path: ev.path,
      owner: ev.owner,
      last_agent: ev.agent,
      review_status: 'unreviewed',
      last_seq: null,
      passes: [],
      open_findings: [],
      closed_findings: [],
      last_reviewed_at: null,
    };

    row.last_seq = ev.seq;
    row.last_agent = ev.agent;
    row.owner = ev.owner;

    if (ev.action === 'review_started') {
      row.review_status = ev.status ?? 'in_progress';
      if (ev.pass != null && !row.passes.includes(ev.pass)) {
        row.passes = [...row.passes, ev.pass].sort((a, b) => a - b);
      }
      row.last_reviewed_at = ev.at;
    } else if (ev.action === 'review_completed') {
      row.review_status = ev.status;
      if (ev.pass != null && !row.passes.includes(ev.pass)) {
        row.passes = [...row.passes, ev.pass].sort((a, b) => a - b);
      }
      row.last_reviewed_at = ev.at;
    } else if (ev.action === 'finding_logged') {
      if (ev.finding_id && !row.open_findings.includes(ev.finding_id)) {
        row.open_findings = [...row.open_findings, ev.finding_id];
      }
      row.review_status = row.review_status === 'unreviewed' ? 'in_progress' : row.review_status;
    } else if (ev.action === 'fix_applied') {
      if (ev.finding_id) {
        row.open_findings = row.open_findings.filter(f => f !== ev.finding_id);
        if (!row.closed_findings.includes(ev.finding_id)) {
          row.closed_findings = [...row.closed_findings, ev.finding_id];
        }
      }
      if (row.open_findings.length === 0 && row.review_status === 'needs_followup') {
        row.review_status = 'fixed_pending_recheck';
      }
    } else if (ev.action === 'rechecked') {
      row.review_status = ev.status ?? row.review_status;
      row.last_reviewed_at = ev.at;
    } else if (ev.action === 'ownership_changed') {
      row.owner = ev.owner;
      row.last_agent = ev.agent;
    }

    byPath.set(ev.path, row);
  }

  const rows = [...byPath.values()].sort((a, b) => a.path.localeCompare(b.path));
  writeJsonl(LATEST_FILE, rows);
  console.log(`Regenerated latest.jsonl (${rows.size ?? rows.length} files)`);
}

// ── event ─────────────────────────────────────────────────────────────────────

const VALID_ACTIONS = new Set([
  'review_started', 'finding_logged', 'fix_applied', 'rechecked', 'review_completed', 'ownership_changed',
]);
const VALID_STATUSES = new Set([
  'unreviewed', 'in_progress', 'reviewed_clean', 'needs_followup', 'fixed_pending_recheck', 'closed',
]);

function appendEvent(args) {
  ensureDir(AUDIT_DIR);

  for (const f of ['path', 'owner', 'agent', 'action']) {
    if (!args[f]) throw new Error(`Missing required event field: --${f}`);
  }
  if (!VALID_ACTIONS.has(args.action)) {
    throw new Error(`Unknown action "${args.action}". Valid: ${[...VALID_ACTIONS].join(', ')}`);
  }
  if (args.status && !VALID_STATUSES.has(args.status)) {
    throw new Error(`Unknown status "${args.status}". Valid: ${[...VALID_STATUSES].join(', ')}`);
  }

  const seq = nextSeq();
  const event = {
    seq,
    path: args.path,
    owner: args.owner,
    agent: args.agent,
    action: args.action,
    ...(args.status && { status: args.status }),
    ...(args.pass != null && { pass: Number(args.pass) }),
    ...(args.findingId && { finding_id: args.findingId }),
    ...(args.severity && { severity: args.severity }),
    ...(args.summary && { summary: args.summary }),
    ...(args.commit && { commit: args.commit }),
    at: args.at || new Date().toISOString(),
  };

  appendFileSync(EVENTS_FILE, JSON.stringify(event) + '\n');
  console.log(`Appended seq=${seq} action=${event.action} path=${event.path}`);
  regenLatest();
}

// ── summary ───────────────────────────────────────────────────────────────────

function summary() {
  const rows = readJsonl(LATEST_FILE);
  const counts = {};
  for (const r of rows) {
    counts[r.review_status] = (counts[r.review_status] ?? 0) + 1;
  }
  const openFindings = rows.reduce((n, r) => n + r.open_findings.length, 0);
  console.log('\nReview summary:');
  for (const [status, n] of Object.entries(counts).sort()) {
    console.log(`  ${status.padEnd(24)} ${n}`);
  }
  console.log(`  ${'open findings'.padEnd(24)} ${openFindings}`);
  console.log(`  ${'total files'.padEnd(24)} ${rows.length}\n`);

  // Per-owner breakdown
  const owners = {};
  for (const r of rows) {
    if (!owners[r.owner]) owners[r.owner] = { total: 0, unreviewed: 0, open: 0 };
    owners[r.owner].total++;
    if (r.review_status === 'unreviewed') owners[r.owner].unreviewed++;
    owners[r.owner].open += r.open_findings.length;
  }
  console.log('Owner breakdown:');
  for (const [owner, o] of Object.entries(owners).sort()) {
    console.log(`  ${owner}  total=${o.total}  unreviewed=${o.unreviewed}  open_findings=${o.open}`);
  }
  console.log();
}

// ── CLI ───────────────────────────────────────────────────────────────────────

function parseArgs(arr) {
  const args = {};
  for (let i = 0; i < arr.length; i++) {
    if (arr[i].startsWith('--')) {
      const raw = arr[i].slice(2);
      const key = raw.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      const next = arr[i + 1];
      args[key] = (!next || next.startsWith('--')) ? true : arr[++i];
    }
  }
  return args;
}

const [,, cmd, ...rest] = process.argv;

if (cmd === 'seed') {
  await seed();
} else if (cmd === 'event') {
  appendEvent(parseArgs(rest));
} else if (cmd === 'regen') {
  regenLatest();
} else if (cmd === 'summary') {
  summary();
} else {
  console.error([
    'Usage: node tools/review-audit.js <command> [options]',
    '',
    'Commands:',
    '  seed                    Populate latest.jsonl from src/ (skips existing entries)',
    '  event                   Append one event and regen latest.jsonl',
    '  regen                   Regenerate latest.jsonl from events.jsonl',
    '  summary                 Print status counts and owner breakdown',
    '',
    'Event options (--path and --owner and --agent and --action are required):',
    '  --path <file>           Relative file path (e.g. src/config/merge.js)',
    '  --owner <A|B|C>         Owner bucket',
    '  --agent <name>          Agent or person performing the action',
    '  --action <action>       review_started | finding_logged | fix_applied |',
    '                          rechecked | review_completed | ownership_changed',
    '  --status <status>       unreviewed | in_progress | reviewed_clean |',
    '                          needs_followup | fixed_pending_recheck | closed',
    '  --pass <n>              Review pass number (integer)',
    '  --finding-id <id>       Finding identifier (e.g. model-load-001)',
    '  --severity <level>      high | medium | low',
    '  --summary <text>        Short description',
    '  --commit <sha>          Commit that applied a fix',
    '  --at <iso8601>          Override timestamp',
  ].join('\n'));
  process.exit(1);
}
