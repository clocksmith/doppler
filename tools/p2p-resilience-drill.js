#!/usr/bin/env node

import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { downloadShard } from '../src/experimental/distribution/shard-delivery.js';

const HTTP_BYTES = new Uint8Array([1, 2, 3, 4]);
const P2P_BYTES = new Uint8Array([9, 10, 11, 12]);
const HTTP_HASH_SHA256 = '9f64a747e1b97f131fabb6b447296c9b6f0201e79fb3c5356e6c77e89b6a806a';
const P2P_HASH_SHA256 = 'e1e853684a206f162ee800a54b695c9cc1a8d1d554a47fcb13fe51229c17773f';
const MANIFEST_VERSION_SET = 'manifest:v1:sha256:drill';
const VALID_STAGES = new Set(['canary', 'regional', 'global']);

function parseArgs(argv) {
  const args = {
    stage: 'canary',
    out: null,
    json: false,
    failOnError: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const token = String(argv[i] ?? '');
    const nextValue = () => {
      const value = argv[i + 1];
      if (value == null || String(value).startsWith('--')) {
        throw new Error(`Missing value for ${token}`);
      }
      i += 1;
      return String(value);
    };
    if (token === '--help' || token === '-h') {
      args.help = true;
      continue;
    }
    if (token === '--stage') {
      args.stage = nextValue();
      continue;
    }
    if (token.startsWith('--stage=')) {
      args.stage = token.split('=', 2)[1];
      continue;
    }
    if (token === '--out') {
      args.out = resolve(process.cwd(), nextValue());
      continue;
    }
    if (token.startsWith('--out=')) {
      args.out = resolve(process.cwd(), token.split('=', 2)[1]);
      continue;
    }
    if (token === '--json') {
      args.json = true;
      continue;
    }
    if (token === '--fail-on-error') {
      args.failOnError = true;
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }

  if (!VALID_STAGES.has(args.stage)) {
    throw new Error(`--stage must be one of: ${[...VALID_STAGES].join(', ')}, got "${args.stage}"`);
  }

  return args;
}

function usage() {
  return [
    'Usage:',
    '  node tools/p2p-resilience-drill.js [--stage <canary|regional|global>] [--out <report.json>] [--json] [--fail-on-error]',
  ].join('\n');
}

function createHttpResponse(payload, options = {}) {
  return new Response(payload, {
    status: Number.isInteger(options.status) ? options.status : 200,
    headers: options.headers ?? {},
  });
}

async function runScenario(definition) {
  const startedAtMs = Date.now();
  try {
    const result = await definition.run();
    const source = result?.source ?? null;
    const passed = source === definition.expectedSource;
    return {
      id: definition.id,
      description: definition.description,
      status: passed ? 'pass' : 'fail',
      expectedSource: definition.expectedSource,
      actualSource: source,
      durationMs: Date.now() - startedAtMs,
      failureCodes: result?.deliveryMetrics?.failureCodes ?? {},
      deliveryMetrics: result?.deliveryMetrics ?? null,
      error: passed ? null : `expected source ${definition.expectedSource}, got ${source}`,
    };
  } catch (error) {
    return {
      id: definition.id,
      description: definition.description,
      status: 'fail',
      expectedSource: definition.expectedSource,
      actualSource: null,
      durationMs: Date.now() - startedAtMs,
      failureCodes: {},
      deliveryMetrics: null,
      error: error?.message || String(error),
    };
  }
}

async function runDrills(stage) {
  const shardInfo = {
    filename: 'shard_0.bin',
    size: 4,
    hash: HTTP_HASH_SHA256,
  };

  const originalFetch = globalThis.fetch;
  try {
    const scenarios = [
      {
        id: 'origin_outage_p2p_survives',
        description: 'Origin outage with healthy peers should continue via P2P path.',
        expectedSource: 'p2p',
        run: async () => {
          globalThis.fetch = async () => createHttpResponse('origin unavailable', {
            status: 503,
          });
          return downloadShard('https://example.com/model', 0, shardInfo, {
            algorithm: 'sha256',
            expectedHash: P2P_HASH_SHA256,
            expectedManifestVersionSet: MANIFEST_VERSION_SET,
            distributionConfig: {
              sourceOrder: ['p2p', 'http'],
              p2p: {
                enabled: true,
                timeoutMs: 1000,
                maxRetries: 0,
                transport: async () => ({
                  data: P2P_BYTES,
                  manifestVersionSet: MANIFEST_VERSION_SET,
                }),
              },
            },
            writeToStore: false,
          });
        },
      },
      {
        id: 'peer_collapse_http_fallback',
        description: 'Peer collapse should degrade to HTTP with deterministic fallback policy.',
        expectedSource: 'http',
        run: async () => {
          globalThis.fetch = async () => createHttpResponse(HTTP_BYTES, {
            status: 200,
          });
          return downloadShard('https://example.com/model', 1, shardInfo, {
            algorithm: 'sha256',
            expectedHash: HTTP_HASH_SHA256,
            expectedManifestVersionSet: MANIFEST_VERSION_SET,
            distributionConfig: {
              sourceOrder: ['p2p', 'http'],
              p2p: {
                enabled: true,
                timeoutMs: 1000,
                maxRetries: 0,
                transport: async () => {
                  throw new Error('peer miss');
                },
              },
            },
            writeToStore: false,
          });
        },
      },
      {
        id: 'anti_rollback_incident_http_recovery',
        description: 'Anti-rollback incident from P2P should be detected and recovered via HTTP.',
        expectedSource: 'http',
        run: async () => {
          globalThis.fetch = async () => createHttpResponse(HTTP_BYTES, {
            status: 200,
          });
          return downloadShard('https://example.com/model', 2, shardInfo, {
            algorithm: 'sha256',
            expectedHash: HTTP_HASH_SHA256,
            expectedManifestVersionSet: MANIFEST_VERSION_SET,
            distributionConfig: {
              sourceOrder: ['p2p', 'http'],
              antiRollback: {
                enabled: true,
                requireExpectedHash: true,
                requireExpectedSize: false,
                requireManifestVersionSet: true,
              },
              p2p: {
                enabled: true,
                timeoutMs: 1000,
                maxRetries: 0,
                transport: async () => ({
                  data: HTTP_BYTES,
                  manifestVersionSet: 'manifest:v0:rollback',
                }),
              },
            },
            writeToStore: false,
          });
        },
      },
    ];

    const results = [];
    for (const scenario of scenarios) {
      // eslint-disable-next-line no-await-in-loop
      const result = await runScenario(scenario);
      results.push(result);
    }

    const passed = results.filter((entry) => entry.status === 'pass').length;
    const failed = results.length - passed;

    return {
      schemaVersion: 1,
      generatedAtMs: Date.now(),
      stage,
      summary: {
        total: results.length,
        passed,
        failed,
      },
      scenarios: results,
    };
  } finally {
    globalThis.fetch = originalFetch;
  }
}

export async function runP2PResilienceDrillCli(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.help) {
    console.log(usage());
    return null;
  }

  if (!VALID_STAGES.has(args.stage)) {
    throw new Error(`--stage must be one of: ${[...VALID_STAGES].join(', ')}, got "${args.stage}"`);
  }

  const report = await runDrills(args.stage);
  const reportJson = `${JSON.stringify(report, null, 2)}\n`;

  if (args.json) {
    console.log(reportJson.trimEnd());
  }

  if (!args.json || args.out) {
    const defaultPath = resolve(
      process.cwd(),
      'reports/p2p-resilience-drills',
      `p2p-resilience-drill-${Date.now()}-${args.stage}.json`
    );
    const reportPath = args.out || defaultPath;
    mkdirSync(resolve(reportPath, '..'), { recursive: true });
    writeFileSync(reportPath, reportJson, 'utf8');
    console.log(`[p2p-drill] report: ${reportPath}`);
  }

  if (args.failOnError && report.summary.failed > 0) {
    throw new Error(`P2P resilience drill failed ${report.summary.failed} scenarios.`);
  }

  return report;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    await runP2PResilienceDrillCli(process.argv.slice(2));
  } catch (error) {
    console.error(`[p2p-drill] ${error?.message || String(error)}`);
    process.exit(1);
  }
}
