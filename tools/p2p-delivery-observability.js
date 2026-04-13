#!/usr/bin/env node

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  aggregateP2PDeliveryObservability,
  buildP2PAlertsFromSummary,
  buildP2PDashboardSnapshot,
} from '../src/experimental/distribution/p2p-observability.js';

function parseRatio(rawValue, flag) {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    throw new Error(`${flag} must be a finite number in [0, 1], got ${JSON.stringify(rawValue)}`);
  }
  return parsed;
}

function parsePositiveMs(rawValue, flag) {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive finite number, got ${JSON.stringify(rawValue)}`);
  }
  return parsed;
}

function parseArgs(argv) {
  const args = {
    input: null,
    outDir: resolve(process.cwd(), 'reports/p2p-observability'),
    json: false,
    targets: {},
  };

  for (let i = 0; i < argv.length; i += 1) {
    const token = String(argv[i] ?? '');
    const nextValue = () => {
      const value = argv[i + 1];
      if (value == null || String(value).startsWith('--')) {
        throw new Error(`Missing value for ${token}`);
      }
      i += 1;
      return value;
    };
    if (token === '--help' || token === '-h') {
      args.help = true;
      continue;
    }
    if (token === '--input') {
      args.input = resolve(process.cwd(), nextValue());
      continue;
    }
    if (token.startsWith('--input=')) {
      args.input = resolve(process.cwd(), token.split('=', 2)[1]);
      continue;
    }
    if (token === '--out-dir') {
      args.outDir = resolve(process.cwd(), nextValue());
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
      args.targets.minAvailability = parseRatio(nextValue(), '--min-availability');
      continue;
    }
    if (token.startsWith('--min-availability=')) {
      args.targets.minAvailability = parseRatio(token.split('=', 2)[1], '--min-availability');
      continue;
    }
    if (token === '--min-p2p-hit-rate') {
      args.targets.minP2PHitRate = parseRatio(nextValue(), '--min-p2p-hit-rate');
      continue;
    }
    if (token.startsWith('--min-p2p-hit-rate=')) {
      args.targets.minP2PHitRate = parseRatio(token.split('=', 2)[1], '--min-p2p-hit-rate');
      continue;
    }
    if (token === '--max-http-fallback-rate') {
      args.targets.maxHttpFallbackRate = parseRatio(nextValue(), '--max-http-fallback-rate');
      continue;
    }
    if (token.startsWith('--max-http-fallback-rate=')) {
      args.targets.maxHttpFallbackRate = parseRatio(token.split('=', 2)[1], '--max-http-fallback-rate');
      continue;
    }
    if (token === '--max-p95-latency-ms') {
      args.targets.maxP95LatencyMs = parsePositiveMs(nextValue(), '--max-p95-latency-ms');
      continue;
    }
    if (token.startsWith('--max-p95-latency-ms=')) {
      args.targets.maxP95LatencyMs = parsePositiveMs(token.split('=', 2)[1], '--max-p95-latency-ms');
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }

  return args;
}

function usage() {
  return [
    'Usage:',
    '  node tools/p2p-delivery-observability.js --input <metrics.json|metrics.ndjson> [--out-dir <dir>] [--json]',
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
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    return parseNdjson(text, inputPath);
  }
  if (Array.isArray(parsed)) {
    return parsed;
  }
  if (parsed && typeof parsed === 'object') {
    if (Array.isArray(parsed.records)) {
      return parsed.records;
    }
    return [parsed];
  }
  throw new Error(`Input JSON at ${inputPath} must be an object, an array, or an object with records[].`);
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
