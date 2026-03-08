#!/usr/bin/env node
// tools/review-audit.js
// Append-only event log + materialized latest-state tracker for repo review scopes.
//
// Subcommands:
//   seed    Walk the selected scope and write initial latest.jsonl rows
//   event   Append one event to events.jsonl, then regen latest.jsonl
//   regen   Regenerate latest.jsonl from events.jsonl
//
// Files:
//   reports/review/<scope>-audit/events.jsonl  -- append-only, one line per action
//   reports/review/<scope>-audit/latest.jsonl  -- one row per file, rewritten on regen

import { appendFileSync, existsSync, mkdirSync } from 'fs';

import {
  assignOwner,
  getAuditPaths,
  getScopeConfig,
  readJsonl,
  toRelativePath,
  validateEventSequence,
  walkScope,
  writeJsonl,
} from './review-audit-lib.js';

function ensureDir(dir) {
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

function nextSeq(eventsFile) {
  const events = readJsonl(eventsFile);
  validateEventSequence(events, eventsFile);
  return events.length === 0 ? 1 : events[events.length - 1].seq + 1;
}

// ── seed ──────────────────────────────────────────────────────────────────────

async function seed(scopeName) {
  const { auditDir, latestFile } = getAuditPaths(scopeName);
  ensureDir(auditDir);
  const allFiles = (await walkScope(scopeName)).sort();

  const existing = new Set(readJsonl(latestFile).map(r => r.path));
  const newRows = [];

  for (const abs of allFiles) {
    const rel = toRelativePath(abs);
    if (existing.has(rel)) continue;
    const { owner, agent } = assignOwner(scopeName, rel);
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
    console.log(`All ${getScopeConfig(scopeName).name}/ files already present in latest.jsonl.`);
    return;
  }

  const all = [...readJsonl(latestFile), ...newRows]
    .sort((a, b) => a.path.localeCompare(b.path));
  writeJsonl(latestFile, all);
  console.log(`Seeded ${newRows.length} file(s) → ${latestFile}`);
}

// ── regen ─────────────────────────────────────────────────────────────────────

function regenLatest(scopeName) {
  const { auditDir, eventsFile, latestFile } = getAuditPaths(scopeName);
  ensureDir(auditDir);
  const events = readJsonl(eventsFile);
  validateEventSequence(events, eventsFile);

  // Start from whatever is already in latest (preserves seeded rows with no events yet)
  const byPath = new Map(readJsonl(latestFile).map(r => [r.path, { ...r }]));

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
      row.review_status = 'needs_followup';
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
  writeJsonl(latestFile, rows);
  console.log(`Regenerated latest.jsonl (${rows.length} files)`);
}

// ── event ─────────────────────────────────────────────────────────────────────

const VALID_ACTIONS = new Set([
  'review_started', 'finding_logged', 'fix_applied', 'rechecked', 'review_completed', 'ownership_changed',
]);
const VALID_STATUSES = new Set([
  'unreviewed', 'in_progress', 'reviewed_clean', 'needs_followup', 'fixed_pending_recheck', 'closed',
]);

async function appendEvent(scopeName, args) {
  const { auditDir, eventsFile, latestFile } = getAuditPaths(scopeName);
  ensureDir(auditDir);

  for (const f of ['path', 'owner', 'agent', 'action']) {
    if (!args[f]) throw new Error(`Missing required event field: --${f}`);
  }
  if (!VALID_ACTIONS.has(args.action)) {
    throw new Error(`Unknown action "${args.action}". Valid: ${[...VALID_ACTIONS].join(', ')}`);
  }
  if (args.status && !VALID_STATUSES.has(args.status)) {
    throw new Error(`Unknown status "${args.status}". Valid: ${[...VALID_STATUSES].join(', ')}`);
  }
  if (args.at && Number.isNaN(Date.parse(args.at))) {
    throw new Error(`--at must be a valid ISO 8601 timestamp; got "${args.at}"`);
  }

  const existingRows = new Map(readJsonl(latestFile).map(row => [row.path, row]));
  const knownPath = existingRows.get(args.path);
  if (!knownPath) {
    const scopePaths = new Set((await walkScope(scopeName)).map(toRelativePath));
    if (!scopePaths.has(args.path)) {
      throw new Error(
        `Path "${args.path}" is not tracked by scope "${scopeName}". Seed the scope first or use a valid path.`
      );
    }
  }
  const canonicalOwner = knownPath?.owner || assignOwner(scopeName, args.path).owner;
  if (args.owner !== canonicalOwner) {
    throw new Error(
      `Owner mismatch for "${args.path}": got "${args.owner}", expected "${canonicalOwner}".`
    );
  }

  const seq = nextSeq(eventsFile);
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

  appendFileSync(eventsFile, JSON.stringify(event) + '\n');
  console.log(`Appended seq=${seq} action=${event.action} path=${event.path}`);
  regenLatest(scopeName);
}

// ── summary ───────────────────────────────────────────────────────────────────

function summary(scopeName) {
  const { latestFile } = getAuditPaths(scopeName);
  const rows = readJsonl(latestFile);
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
    if (!arr[i].startsWith('--')) {
      throw new Error(`Unexpected positional argument: ${arr[i]}`);
    }
    const raw = arr[i].slice(2);
    const key = raw.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    const next = arr[i + 1];
    args[key] = (!next || next.startsWith('--')) ? true : arr[++i];
  }
  return args;
}

const [,, cmd, ...rest] = process.argv;
const args = parseArgs(rest);
const scopeName = args.scope || 'src';

if (cmd === 'seed') {
  await seed(scopeName);
} else if (cmd === 'event') {
  await appendEvent(scopeName, args);
} else if (cmd === 'regen') {
  regenLatest(scopeName);
} else if (cmd === 'summary') {
  summary(scopeName);
} else {
  console.error([
    'Usage: node tools/review-audit.js <command> [options]',
    '',
    'Commands:',
    '  seed                    Populate latest.jsonl from the selected scope (skips existing entries)',
    '  event                   Append one event and regen latest.jsonl',
    '  regen                   Regenerate latest.jsonl from events.jsonl',
    '  summary                 Print status counts and owner breakdown',
    '',
    'General options:',
    '  --scope <src|tools>     Audit scope (default: src)',
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
