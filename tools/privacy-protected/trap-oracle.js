// trap-oracle: secret, fresh known-answer traps per session (trap verification v2). Run by a party the evaluator
// cannot see into (the model publisher or a separate host): it holds the public pinned weights and answers each
// request with a new batch of trap inputs, sampled like real prompts, and their expected outputs from the pinned
// approximate circuit, encrypted to the requester's one-time key and signed with the oracle's pinned Ed25519 key
// (traps.js). Requesters are anonymous to it.
//
// What the oracle learns per request: that a session is starting, its prompt length, generated count and number of
// traps, the time, and the network address the request came from. Never the prompt, the answer or the evaluator.
//
//   node tools/privacy-protected/trap-oracle.js --key oracle.key [--new-key] [--model tiny-model.json]
//        [--listen 127.0.0.1:0] [--ready-file f] [--tolerance 0.75] [--min-margin 2.5] [--ttl 10m] [--exit-with-stdin]

import { writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { randomInt } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { DEFAULT_MODEL, circuitDigest, loadModel, newPlan } from './plan.js';
import { evaluatePlaintext, loadTriageModel } from './triage-model.js';
import { GENERATED, TASK_ID } from './triage-task.js';
import {
  MAX_TRAPS_PER_BATCH, TRAP_DISTRIBUTION, TRAP_REQUEST_LIMIT, b64, issueTrapBatch, loadOrCreateSeedKey, rawPublicKey,
} from './traps.js';

// Go-style durations: 10m, 90s, 1h30m, 500ms.
export function parseDuration(s) {
  const re = /(\d+(?:\.\d+)?)(ms|s|m|h)/g;
  let total = 0;
  let used = 0;
  for (const m of s.matchAll(re)) {
    total += Number(m[1]) * { ms: 1, s: 1e3, m: 6e4, h: 3.6e6 }[m[2]];
    used += m[0].length;
  }
  if (!s || used !== s.length) throw new Error(`invalid duration ${JSON.stringify(s)}`);
  return total;
}

export function createTrapOracle({ pl, modelId, circuit = circuitDigest(pl, modelId), privateKey, tolerance = 0.75, minMargin = 2.5,
  ttlMs = 600000, log = () => {}, sample, distribution = TRAP_DISTRIBUTION }) {
  const oracleKey = b64(rawPublicKey(privateKey));
  const ctx = { pl, modelId, circuit, privateKey, tolerance, minMargin, ttlMs, sample, distribution };
  const handler = (req, res) => {
    const reply = (status, body, type = 'application/json') => {
      res.writeHead(status, { 'Content-Type': type });
      res.end(type === 'application/json' ? `${JSON.stringify(body)}\n` : body);
    };
    if (req.url === '/v1/descriptor') {
      reply(200, { schema: 'doppler.protected-trap-oracle/v1', oracleKey, circuit, model: modelId, distribution,
        maxTraps: MAX_TRAPS_PER_BATCH });
      return;
    }
    if (req.url !== '/v1/trap-batch') {
      reply(404, '404 page not found\n', 'text/plain');
      return;
    }
    if (req.method !== 'POST') {
      reply(405, 'POST only\n', 'text/plain');
      return;
    }
    const chunks = [];
    let size = 0;
    let over = false;
    req.on('data', (c) => {
      size += c.length;
      if (size > TRAP_REQUEST_LIMIT) over = true;
      else chunks.push(c);
    });
    req.on('end', () => {
      let request;
      try {
        if (over) throw new Error('too large');
        request = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      } catch {
        reply(400, 'malformed request\n', 'text/plain');
        return;
      }
      const t0 = performance.now();
      let env;
      try {
        env = issueTrapBatch(ctx, request);
      } catch (e) {
        reply(400, `${e.message}\n`, 'text/plain');
        return;
      }
      // Sizes and timing only: the oracle never has a prompt to log.
      log(`[trap-oracle] batch of ${request.count}, prompt length ${request.promptLength}, generate ${request.generate}, `
        + `${(performance.now() - t0).toFixed(1)} ms`);
      reply(200, env);
    });
  };
  return { circuit, oracleKey, handler, issue: (request, now) => issueTrapBatch(ctx, request, now) };
}

export function createTriageTrapOracle({ modelPath, privateKey, minMargin = 0.2, ...options }) {
  const { model, id: modelId, circuit } = loadTriageModel(modelPath);
  const rows = ['train', 'validation', 'holdout'].flatMap(split => evaluatePlaintext(model, split).rows);
  const candidates = rows.filter(({ approximate }) => !approximate.nonFiniteLogits && !approximate.rangeViolations.length
    && approximate.logits.every(logits => {
      const [first, second] = [...logits].sort((a, b) => b - a);
      return first - second >= minMargin;
    }));
  if (!candidates.length) throw new Error('No eligible known answers for the pinned triage task');
  function sample({ promptLength, generate, count }) {
    if (promptLength !== 2 || generate !== GENERATED) throw new Error('Only the pinned ticket schedule is supported');
    return Array.from({ length: count }, (_, i) => {
      const { tokens, approximate } = candidates[randomInt(candidates.length)];
      return { id: `t${i}`, prompt: tokens, continuation: approximate.output.slice(0, -1), expectedLogits: approximate.logits };
    });
  }
  return createTrapOracle({ ...options, modelId, circuit, privateKey, minMargin, sample, distribution: TASK_ID });
}

export async function main(argv = process.argv.slice(2)) {
  const { values } = parseArgs({
    args: argv,
    options: {
      model: { type: 'string', default: DEFAULT_MODEL.pathname },
      triage: { type: 'boolean' },
      key: { type: 'string' },
      'new-key': { type: 'boolean', default: false },
      listen: { type: 'string', default: '127.0.0.1:0' },
      'ready-file': { type: 'string' },
      tolerance: { type: 'string', default: '0.75' },
      'min-margin': { type: 'string', default: '2.5' },
      ttl: { type: 'string', default: '10m' },
      'exit-with-stdin': { type: 'boolean', default: false },
    },
  });
  if (!values.key) throw new Error('--key is required');
  const privateKey = await loadOrCreateSeedKey(values.key, values['new-key']);
  const options = { privateKey, tolerance: Number(values.tolerance), minMargin: Number(values['min-margin']),
    ttlMs: parseDuration(values.ttl), log: (m) => console.error(m) };
  let oracle;
  if (values.triage) {
    oracle = createTriageTrapOracle({ ...options, modelPath: values.model === DEFAULT_MODEL.pathname ? undefined : values.model });
  } else {
    const { model, id } = loadModel(values.model);
    oracle = createTrapOracle({ ...options, pl: newPlan(model), modelId: id });
  }
  if (values['exit-with-stdin']) {
    process.stdin.on('end', () => process.exit(3));
    process.stdin.on('error', () => process.exit(3));
    process.stdin.resume();
  }
  const [host, port] = values.listen.split(/:(?=[^:]*$)/);
  const server = createServer(oracle.handler);
  server.headersTimeout = 10000;
  await new Promise((resolve) => server.listen(Number(port), host, resolve));
  const addr = server.address();
  if (values['ready-file']) writeFileSync(values['ready-file'], `${addr.address}:${addr.port}`, { mode: 0o600 });
  console.error(`[trap-oracle] circuit ${oracle.circuit}, key ${oracle.oracleKey}, listening on ${addr.address}:${addr.port}`);
  return server;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => {
    console.error(`error: ${e.message}`);
    process.exit(1);
  });
}
