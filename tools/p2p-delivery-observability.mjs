#!/usr/bin/env node

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  aggregateP2PDeliveryObservability,
  buildP2PAlertsFromSummary,
  buildP2PDashboardSnapshot,
} from '../src/distribution/p2p-observability.js';

function parseArgs(argv) {
  const args = {
    input: null,
    outDir: resolve(process.cwd(), 'reports/p2p-observability'),
    json: false,
    targets: {},
  };

  for (let i = 0; i < argv.length; i += 1) {
    const token = String(argv[i] ?? '');
    if (token === '--help' || token === '-h') {
      args.help = true;
      continue;
    }
    if (token === '--input') {
      args.input = argv[i + 1] ? resolve(process.cwd(), argv[i + 1]) : null;
      i += 1;
      continue;
    }
    if (token.startsWith('--input=')) {
      args.input = resolve(process.cwd(), token.split('=', 2)[1]);
      continue;
    }
    if (token === '--out-dir') {
      args.outDir = argv[i + 1] ? resolve(process.cwd(), argv[i + 1]) : args.outDir;
      i += 1;
      continue;
    }
    if (token.startsWith('--out-dir=')) {
      args.outDir = resolve(process.cwd(), token.split('=', 2)[1]);
      continue;
    }
    if (token === '--json') {
      args.json = true;
      continue;
    }
    if (token === '--min-availability') {
      args.targets.minAvailability = Number(argv[i + 1]);
      i += 1;
      continue;
    }
    if (token.startsWith('--min-availability=')) {
      args.targets.minAvailability = Number(token.split('=', 2)[1]);
      continue;
    }
    if (token === '--min-p2p-hit-rate') {
      args.targets.minP2PHitRate = Number(argv[i + 1]);
      i += 1;
      continue;
    }
    if (token.startsWith('--min-p2p-hit-rate=')) {
      args.targets.minP2PHitRate = Number(token.split('=', 2)[1]);
      continue;
    }
    if (token === '--max-http-fallback-rate') {
      args.targets.maxHttpFallbackRate = Number(argv[i + 1]);
      i += 1;
      continue;
    }
    if (token.startsWith('--max-http-fallback-rate=')) {
      args.targets.maxHttpFallbackRate = Number(token.split('=', 2)[1]);
      continue;
    }
    if (token === '--max-p95-latency-ms') {
      args.targets.maxP95LatencyMs = Number(argv[i + 1]);
      i += 1;
      continue;
    }
    if (token.startsWith('--max-p95-latency-ms=')) {
      args.targets.maxP95LatencyMs = Number(token.split('=', 2)[1]);
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }

  return args;
}

function usage() {
  return [
    'Usage:',
    '  node tools/p2p-delivery-observability.mjs --input <metrics.json|metrics.ndjson> [--out-dir <dir>] [--json]',
    '  [--min-availability <ratio>] [--min-p2p-hit-rate <ratio>] [--max-http-fallback-rate <ratio>] [--max-p95-latency-ms <ms>]',
  ].join('\n');
}

function parseNdjson(text, sourcePath) {
  const rows = [];
  const lines = text.split(/\r?\n/u);
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line) continue;
    try {
      rows.push(JSON.parse(line));
    } catch (error) {
      throw new Error(`Invalid NDJSON at ${sourcePath}:${i + 1}: ${error.message}`);
    }
  }
  return rows;
}

function loadInputRecords(inputPath) {
  const text = readFileSync(inputPath, 'utf8');
  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) {
      return parsed;
    }
    if (parsed && typeof parsed === 'object') {
      if (Array.isArray(parsed.records)) {
        return parsed.records;
      }
      return [parsed];
    }
  } catch {
    return parseNdjson(text, inputPath);
  }
  return [];
}

export function runP2PDeliveryObservabilityCli(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.help) {
    console.log(usage());
    return null;
  }
  if (!args.input) {
    throw new Error('Missing required --input path.');
  }

  const records = loadInputRecords(args.input);
  const snapshot = buildP2PDashboardSnapshot(records, {
    targets: args.targets,
  });
  const summary = aggregateP2PDeliveryObservability(records, {
    targets: args.targets,
  });
  const alerts = buildP2PAlertsFromSummary(summary);

  if (args.json) {
    const output = {
      schemaVersion: snapshot.schemaVersion,
      input: args.input,
      summary,
      alerts,
      snapshot,
    };
    console.log(JSON.stringify(output, null, 2));
    return output;
  }

  mkdirSync(args.outDir, { recursive: true });
  const summaryPath = resolve(args.outDir, 'dashboard-summary.json');
  const alertsPath = resolve(args.outDir, 'alerts.json');
  const sloPath = resolve(args.outDir, 'slo.json');
  const snapshotPath = resolve(args.outDir, 'snapshot.json');

  writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  writeFileSync(alertsPath, `${JSON.stringify(alerts, null, 2)}\n`, 'utf8');
  writeFileSync(sloPath, `${JSON.stringify(summary.slo, null, 2)}\n`, 'utf8');
  writeFileSync(snapshotPath, `${JSON.stringify(snapshot, null, 2)}\n`, 'utf8');

  console.log(`[p2p-observability] records=${summary.totals.records} status=${summary.slo.status}`);
  console.log(`[p2p-observability] summary: ${summaryPath}`);
  console.log(`[p2p-observability] alerts: ${alertsPath}`);
  console.log(`[p2p-observability] slo: ${sloPath}`);
  console.log(`[p2p-observability] snapshot: ${snapshotPath}`);

  return {
    schemaVersion: snapshot.schemaVersion,
    input: args.input,
    summary,
    alerts,
    snapshot,
    outputPaths: {
      summary: summaryPath,
      alerts: alertsPath,
      slo: sloPath,
      snapshot: snapshotPath,
    },
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    runP2PDeliveryObservabilityCli(process.argv.slice(2));
  } catch (error) {
    console.error(`[p2p-observability] ${error?.message || String(error)}`);
    process.exit(1);
  }
}
