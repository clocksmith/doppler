#!/usr/bin/env node
/**
 * Route 3 requester in JavaScript: Qwen 3.5 with every projection evaluated by a stage's two
 * share operators, on 2-party additive shares in Z_(2^64). Runs in Node here; model.js, cluster.js and ring.js are
 * environment-neutral and take their transport and crypto from the caller, so a browser can host the same requester.
 *
 *   node requester.js generate --model DIR --plan PLAN.json --operators OPS.json --prompts PROMPTS.json --out REPORT.json
 *        [--max-tokens 32] [--forced REPORT.json] [--logits-out PREFIX] [--return-shift 24] [--prg aes256-ctr]
 *        [--pipeline-steps] [--explicit] [--keep-rounds]
 *   node requester.js control --model DIR --plan PLAN.json --prompts PROMPTS.json --out REPORT.json [...]   plaintext linears
 *   node requester.js speculate --model DIR --plan PLAN.json (--operators OPS.json | --plain) --workloads W.json --out REPORT.json
 *        [--ks 0,1,4,8] [--only id,id] [--max-tokens 32]
 *        Speculative verification (speculate.js): each prompt is prefilled once, then every k decodes from a copy of that
 *        state with exactly k + 1 columns per pass (prompt-lookup drafts, padded). W.json: [{id, workload, messages}].
 *        --plain uses the plaintext linears (development correctness only).
 *   node requester.js run --model DIR --plan PLAN.json --operators OPS.json --request REQUEST.json --out REPORT.json [...]
 *        One request for a supervisor: a JSON event per line on stdout (ready, token, end or failed); `stop` on stdin
 *        stops after the token in progress. REQUEST.json: {"messages": [{"role": "user", "content": ...}], "maxTokens": N}.
 *        No speculation: one position per step. An operator that misses a reply deadline (--round-timeout-ms per
 *        round and session message, --close-timeout-ms for the close) fails the request with a `failed` event naming
 *        the stalled stage and share (code operator-stalled); every operator connection is closed at once and the
 *        process exits, so a stalled operator cannot hold the request open.
 *   node requester.js bench-shape --shape 27b --operators NULL_OPS.json --out REPORT.json [--prompt-tokens 146]
 *        [--max-tokens 32] [--embedding-file FILE] [...]   requester-only work at a model's shape, against null-operator.js
 *
 * --sharing off  the plaintext control (generate only): the same plan, cut, hosts, encoding, rounds
 *        and links, with each stage's input sent unshared to one operator (--plaintext-share 0|1, default 0, the OPS
 *        entry that is used; the other is not contacted). That operator reads the activations. Refused in `run`, which
 *        serves the secret-shared route.
 *
 * --threads N splits the DeltaNet value heads over N worker threads (default 1: everything on the main thread); the
 * arithmetic per head is the same either way.
 *
 * Every option that changes the protocol (--return-shift, --prg, --pipeline-steps) is off by default, and is used only
 * with operators that advertise it; the report records what each stage actually used.
 */
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { parseArgs } from 'node:util';
import { Tokenizer } from '../../../src/inference/tokenizer.js';
import { SharedCluster } from './cluster.js';
import { QwenRequesterModel, configFrom } from './model.js';
import { fileSource, nodeEnv, npyWriter } from './node.js';
import { createHeadPool } from './parallel.js';
import { PlainLinear } from './plain.js';
import { syntheticConfig, syntheticPlan, syntheticSource } from './synthetic.js';
import { argmaxRows, cloneState, speculativeDecode, summarize } from './speculate.js';

const { positionals, values: args } = parseArgs({ allowPositionals: true, options: {
  model: { type: 'string' }, plan: { type: 'string' }, operators: { type: 'string' }, prompts: { type: 'string' },
  request: { type: 'string' }, out: { type: 'string' }, forced: { type: 'string' }, 'logits-out': { type: 'string' },
  'max-tokens': { type: 'string', default: '32' }, 'return-shift': { type: 'string', default: '0' }, prg: { type: 'string', default: 'shake256' },
  'pipeline-steps': { type: 'boolean' }, 'full-digests': { type: 'boolean' }, threads: { type: 'string', default: '1' }, explicit: { type: 'boolean' }, 'keep-rounds': { type: 'boolean' }, strict: { type: 'boolean' },
  'source-config': { type: 'string' }, shape: { type: 'string' }, 'embedding-file': { type: 'string' }, 'prompt-tokens': { type: 'string', default: '146' }, 'exit-operators': { type: 'boolean' }, 'prompt-limit': { type: 'string' },
  sharing: { type: 'string', default: 'on' }, 'plaintext-share': { type: 'string', default: '0' },
  'round-timeout-ms': { type: 'string', default: '300000' }, 'close-timeout-ms': { type: 'string', default: '5000' },
  workloads: { type: 'string' }, ks: { type: 'string', default: '0,1,4,8' }, only: { type: 'string' }, plain: { type: 'boolean' } } });
const command = positionals[0];
const readJson = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const cpuMs = () => { const u = process.cpuUsage(); return (u.user + u.system) / 1e3; };
const peakRss = () => process.resourceUsage().maxRSS * 1024;
const pool = Number(args.threads) > 1 ? createHeadPool(Number(args.threads)) : null;
const template = prompt => `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
/** The same template for a message list (system + user), thinking off. */
const render = messages => messages.map(m => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join('') + '<|im_start|>assistant\n<think>\n\n</think>\n\n';

async function openModel(linear) {
  const source = fileSource(args.model);
  const config = configFrom(source.manifest, readJson(args['source-config'] ?? path.join(args.model, 'source-config.json')));
  const tokenizer = new Tokenizer();
  await tokenizer.initialize(source.manifest, { loadTokenizerJson: async () => readJson(path.join(args.model, 'tokenizer.json')) });
  return { source, config, tokenizer, model: new QwenRequesterModel(config, source, linear, { parallel: pool }) };
}

async function openCluster(plan) {
  if (!['on', 'off'].includes(args.sharing)) throw Error('--sharing on or off');
  return SharedCluster.open(plan, readJson(args.operators), nodeEnv, { sharing: args.sharing === 'on', plaintextShare: Number(args['plaintext-share']),
    seeded: !args.explicit, returnShift: Number(args['return-shift']), prg: args.prg, pipelineSteps: args['pipeline-steps'], strict: args.strict, fullDigests: args['full-digests'],
    roundTimeoutMs: Number(args['round-timeout-ms']), closeTimeoutMs: Number(args['close-timeout-ms']) });
}

/** Greedy decode (or teacher-forced on `forced`): one prefill pass, then one position per step. */
async function greedy(model, config, ids, maxTokens, { cluster = null, forced = null, onToken = null, shouldStop = null, logits = null } = {}) {
  const state = model.newState();
  if (forced) maxTokens = Math.min(maxTokens, forced.length);
  const tokens = [], steps = [];
  let inputs = ids;
  const base = cluster ? (cluster.step === null ? 0 : cluster.step + 1) : 0;
  for (let step = 0; step < maxTokens; step++) {
    if (cluster) cluster.step = base + step;
    const before = cluster ? cluster.rounds.length : 0;
    const bytes0 = cluster?.bytes();
    const t0 = performance.now(), c0 = cpuMs();
    const row = await model.forward(state, inputs, { keep: 1 });
    const wall = performance.now() - t0, cpu = cpuMs() - c0;
    let top = 0, second = -1;
    for (let i = 1; i < row.length; i++) {
      if (row[i] > row[top]) { second = top; top = i; } else if (second < 0 || row[i] > row[second]) second = i;
    }
    const entry = { step, columns: inputs.length, wallMs: wall, cpuMs: cpu, token: top, margin: row[top] - row[second] };
    if (cluster) {
      const rounds = cluster.rounds.slice(before), bytes1 = cluster.bytes();
      Object.assign(entry, { rounds: rounds.length, up: rounds.reduce((s, r) => s + r.up[0] + r.up[1], 0), down: rounds.reduce((s, r) => s + r.down[0] + r.down[1], 0),
        upAll: bytes1.sent - bytes0.sent, downAll: bytes1.received - bytes0.received,
        wireMs: rounds.reduce((s, r) => s + r.wireMs, 0), operatorMs: rounds.reduce((s, r) => s + Math.max(...r.operatorMs), 0),
        encodeMs: rounds.reduce((s, r) => s + r.encodeMs, 0), decodeMs: rounds.reduce((s, r) => s + r.decodeMs, 0),
        replyMs: [0, 1].map(i => rounds.reduce((s, r) => s + r.replyMs[i], 0)) });
    }
    steps.push(entry);
    logits?.push(row);
    tokens.push(top);
    const done = !forced && (config.eos.includes(top) || step + 1 === maxTokens);
    if (cluster) {
      const e0 = performance.now(), ec = cpuMs();
      await cluster.endStep(base + step);
      entry.endStepMs = performance.now() - e0; entry.wallMs += entry.endStepMs; entry.cpuMs += cpuMs() - ec;
    }
    onToken?.(step, top, entry, done);
    if (!forced && config.eos.includes(top)) break;
    if (shouldStop?.()) { entry.stoppedAfter = true; break; }
    inputs = [forced ? forced[step] : top];
  }
  return { tokens, steps };
}

async function runPrompts(model, config, tokenizer, cluster) {
  const prompts = readJson(args.prompts).slice(0, args['prompt-limit'] ? Number(args['prompt-limit']) : undefined);
  const forced = args.forced ? readJson(args.forced).results : null;
  const results = [];
  for (const [index, prompt] of prompts.entries()) {
    const ids = tokenizer.encode(template(prompt));
    const writer = args['logits-out'] ? npyWriter(`${args['logits-out']}-${index}.npy`, config.vocab) : null;
    const t0 = performance.now();
    const { tokens, steps } = await greedy(model, config, ids, Number(args['max-tokens']), { cluster, forced: forced?.[index]?.tokens, logits: writer });
    writer?.close();
    results.push({ prompt, promptTokens: ids.length, promptIds: ids, tokens, text: tokenizer.decode(tokens, false, false), steps, wallS: (performance.now() - t0) / 1e3 });
    console.error(JSON.stringify({ prompt: index, tokens: tokens.length, wallS: +results.at(-1).wallS.toFixed(2), text: results.at(-1).text.slice(0, 80) }));
  }
  return results;
}

function freshness(cluster) {
  if (!cluster.sharing) return null;
  const digests = cluster.rounds.map(r => r.shareBSha256), counters = cluster.rounds.map(r => r.counter);
  return { rounds: digests.length, distinctShareBDigests: new Set(digests).size, countersStrictlyIncreasing: counters.every((c, i) => i === 0 || c > counters[i - 1]),
    digestOf: cluster.digestLimit ? `the first ${cluster.digestLimit} bytes of each share B (uniform, so distinct prefixes show distinct shares)` : 'every byte of each share B' };
}

async function cmdGenerate() {
  const plan = readJson(args.plan);
  const t0 = performance.now();
  const cluster = await openCluster(plan);
  const { model, config, tokenizer, source } = await openModel(cluster);
  const loadS = (performance.now() - t0) / 1e3;
  const results = await runPrompts(model, config, tokenizer, cluster);
  const stats = await cluster.close(args['exit-operators'] ? 'exit' : 'close');
  const report = { schema: 'doppler.secret-shared-generation/v1', requester: 'javascript', requesterThreads: pool?.threads ?? 1, node: process.version, modelId: plan.modelId, fx: plan.fx, fw: plan.fw,
    seeded: cluster.seeded, sharing: cluster.sharing ? 'on' : 'off', speculation: 'off',
    protocol: { stages: cluster.describe(), pipelineSteps: Boolean(args['pipeline-steps']), roundTimeoutMs: cluster.roundTimeoutMs }, teacherForced: Boolean(args.forced), loadS,
    requesterHolds: { tensors: source.heldTensors(), bytes: source.heldBytes(), embeddingBytesResident: 0, projectionBytes: 0 },
    results, freshness: freshness(cluster), operatorStats: stats, requesterPeakRss: peakRss(), rounds: args['keep-rounds'] ? cluster.rounds : null };
  fs.writeFileSync(args.out, JSON.stringify(report) + '\n');
  console.error(JSON.stringify({ freshness: report.freshness, requesterPeakRss: report.requesterPeakRss, protocol: report.protocol }));
}

async function cmdControl() {
  const plan = readJson(args.plan);
  const source = fileSource(args.model);
  const { model, config, tokenizer } = await openModel(new PlainLinear(plan, source));
  const results = await runPrompts(model, config, tokenizer, null);
  fs.writeFileSync(args.out, JSON.stringify({ schema: 'doppler.secret-shared-control/v1', requester: 'javascript', teacherForced: Boolean(args.forced), results, requesterPeakRss: peakRss() }) + '\n');
}

async function cmdRun() {
  const emit = event => process.stdout.write(JSON.stringify(event) + '\n');
  let stop = false;
  readline.createInterface({ input: process.stdin }).on('line', line => { if (line.trim() === 'stop') stop = true; });
  if (args.sharing !== 'on') { emit({ event: 'failed', error: 'run serves the secret-shared route: --sharing off is for generate only' }); process.exit(2); }
  const request = readJson(args.request);
  if (request.messages?.length !== 1 || request.messages[0].role !== 'user') throw Error('one user message');
  let cluster = null, lastEventAt = performance.now();
  const t0 = performance.now();
  try {
    const plan = readJson(args.plan);
    cluster = await openCluster(plan);
    const { model, config, tokenizer, source } = await openModel(cluster);
    const ids = tokenizer.encode(template(request.messages[0].content));
    emit({ event: 'ready', promptTokens: ids.length, loadS: (performance.now() - t0) / 1e3,
      stages: cluster.pairs.flat().map(op => [op.hello.stage, op.hello.share, op.hello.planSha256]) });
    const generated = [];
    let text = '';
    const { tokens, steps } = await greedy(model, config, ids, request.maxTokens, { cluster, shouldStop: () => stop, onToken: (step, token, row, done) => {
      generated.push(token);
      const full = tokenizer.decode(generated, true, false);
      const delta = full.slice(text.length); text = full;
      const reason = config.eos.includes(token) ? 'eos' : done ? 'max-tokens' : null;
      lastEventAt = performance.now();
      emit({ event: 'token', step, tokenId: token, delta, done: Boolean(done), stopReason: reason, inputTokens: row.columns,
        ...Object.fromEntries(['wallMs', 'cpuMs', 'rounds', 'up', 'down', 'wireMs', 'operatorMs', 'margin'].map(k => [k, row[k] ?? null])) });
    } });
    const stopped = Boolean(steps.at(-1)?.stoppedAfter);
    const stats = await cluster.close();
    const report = { schema: 'doppler.secret-shared-run/v1', requester: 'javascript', modelId: plan.modelId, fx: plan.fx, fw: plan.fw,
      protocol: { stages: cluster.describe(), pipelineSteps: Boolean(args['pipeline-steps']), speculation: 'off', roundTimeoutMs: cluster.roundTimeoutMs },
      requesterHolds: { tensors: source.heldTensors(), bytes: source.heldBytes(), projectionBytes: 0 },
      promptTokens: ids.length, tokens, text, steps, freshness: freshness(cluster), operatorStats: stats, requesterPeakRss: peakRss(), wallS: (performance.now() - t0) / 1e3 };
    fs.writeFileSync(args.out, JSON.stringify(report) + '\n');
    emit({ event: 'end', stopReason: stopped ? 'requester-stop' : tokens.length && config.eos.includes(tokens.at(-1)) ? 'eos' : 'max-tokens',
      outputTokens: tokens.length, freshness: report.freshness, requesterPeakRss: report.requesterPeakRss });
    process.exit(0);
  } catch (error) {
    // Cancel first: every operator connection closes now, so no operator keeps a round open for this request.
    const cancelled = cluster ? cluster.cancel(error) : null;
    const failedAt = performance.now();
    emit({ event: 'failed', error: `${error?.name}: ${error?.message}`.slice(0, 500), code: error?.code ?? null,
      stalled: error?.stalled ?? (error?.operator ? [error.operator] : null), timeoutMs: error?.timeoutMs ?? null,
      sinceLastEventMs: failedAt - lastEventAt, wallMs: failedAt - t0, cancelled: Boolean(cancelled),
      roundTimeoutMs: Number(args['round-timeout-ms']) });
    process.exit(1);
  }
}

async function cmdBenchShape() {
  const config = syntheticConfig(args.shape);
  const plan = syntheticPlan(config, readJson(args.operators).stages.length);
  const cluster = await openCluster(plan);
  const source = syntheticSource(config, args['embedding-file']);
  const model = new QwenRequesterModel(config, source, cluster, { parallel: pool });
  const ids = Array.from({ length: Number(args['prompt-tokens']) }, (_, i) => (i * 7919) % config.vocab);
  const forced = Array.from({ length: Number(args['max-tokens']) }, (_, i) => (i * 104729) % config.vocab);
  const { steps } = await greedy(model, config, ids, forced.length, { cluster, forced });
  await cluster.close();
  const decode = steps.slice(1), med = key => decode.map(s => s[key]).sort((a, b) => a - b)[decode.length >> 1];
  const report = { schema: 'doppler.secret-shared-requester-shape/v1', note: 'requester-only work at the shape, against null operators; not a model run',
    shape: args.shape, requesterThreads: pool?.threads ?? 1, config: { ...config, layerTypes: undefined }, protocol: { stages: cluster.describe(), pipelineSteps: Boolean(args['pipeline-steps']) },
    prefill: steps[0], decodeMedian: { cpuMs: med('cpuMs'), wallMs: med('wallMs'), up: med('up'), down: med('down'), rounds: med('rounds'), encodeMs: med('encodeMs'), decodeMs: med('decodeMs') },
    steps, requesterPeakRss: peakRss(), node: process.version };
  fs.writeFileSync(args.out, JSON.stringify(report) + '\n');
  console.error(JSON.stringify({ prefill: { columns: steps[0].columns, cpuMs: Math.round(steps[0].cpuMs), wallMs: Math.round(steps[0].wallMs) }, decodeMedian: report.decodeMedian, requesterPeakRss: report.requesterPeakRss }));
}

async function cmdSpeculate() {
  const plan = readJson(args.plan);
  const ks = args.ks.split(',').map(Number);
  const cluster = args.plain ? null : await openCluster(plan);
  const { model, config, tokenizer, source } = await openModel(cluster ?? new PlainLinear(plan, fileSource(args.model)));
  let rows = readJson(args.workloads);
  if (args.only) rows = rows.filter(r => args.only.split(',').includes(r.id));
  const maxTokens = Number(args['max-tokens']);
  const meta = { schema: 'doppler.secret-shared-speculation/v1', requester: 'javascript', node: process.version, modelId: plan.modelId,
    fx: plan.fx, fw: plan.fw, linears: args.plain ? 'plaintext (development)' : 'secret-shared', ks,
    protocol: cluster ? { stages: cluster.describe() } : null, draft: 'prompt lookup (3-gram to 1-gram), padded to k' };
  const results = [];
  for (const row of rows) {
    const ids = tokenizer.encode(render(row.messages));
    const state = model.newState();
    const before = cluster ? cluster.rounds.length : 0, bytes0 = cluster?.bytes();
    const t0 = performance.now(), c0 = cpuMs();
    const first = argmaxRows(await model.forward(state, ids, { keep: 1 }), 1, config.vocab)[0];
    const prefill = { columns: ids.length, wallMs: performance.now() - t0, cpuMs: cpuMs() - c0 };
    if (cluster) {
      const rounds = cluster.rounds.slice(before), bytes1 = cluster.bytes();
      Object.assign(prefill, { rounds: rounds.length, up: bytes1.sent - bytes0.sent, down: bytes1.received - bytes0.received,
        wireMs: rounds.reduce((s, r) => s + r.wireMs, 0), operatorMs: rounds.reduce((s, r) => s + Math.max(...r.operatorMs), 0) });
    }
    const result = { id: row.id, workload: row.workload, promptTokens: ids.length, prefill, byK: {} };
    for (const k of ks) {
      const { tokens, passes } = await speculativeDecode(model, cloneState(state), { promptIds: ids, first, k, eos: config.eos, maxTokens, cluster, cpuMs });
      const summary = summarize(tokens, passes);
      result.byK[k] = { tokens, text: tokenizer.decode(tokens, false, false), passes, summary };
      console.error(JSON.stringify({ id: row.id, k, tokens: tokens.length, passes: passes.length, sPerToken: summary && +summary.sPerToken.toFixed(3),
        hist: summary?.acceptedHistogram, same: tokens.join() === result.byK[ks[0]].tokens.join() }));
    }
    results.push(result);
    fs.writeFileSync(args.out, JSON.stringify({ ...meta, partial: true, results }) + '\n');
  }
  const mismatches = results.flatMap(r => ks.filter(k => r.byK[k].tokens.join() !== r.byK[ks[0]].tokens.join()).map(k => ({ id: r.id, k })));
  const stats = cluster ? await cluster.close(args['exit-operators'] ? 'exit' : 'close') : null;
  const report = { ...meta, results, correctness: { prompts: results.length, identicalAcrossK: mismatches.length === 0, mismatches },
    freshness: cluster ? freshness(cluster) : null, operatorStats: stats, requesterPeakRss: peakRss(),
    requesterHolds: { tensors: source.heldTensors(), bytes: source.heldBytes() } };
  fs.writeFileSync(args.out, JSON.stringify(report) + '\n');
  console.error(JSON.stringify({ correctness: report.correctness, freshness: report.freshness }));
}

const commands = { generate: cmdGenerate, control: cmdControl, run: cmdRun, 'bench-shape': cmdBenchShape, speculate: cmdSpeculate };
if (!commands[command] || (!args.model && command !== 'bench-shape') || (command !== 'run' && !args.out)) {
  console.error('usage: requester.js generate|control|run --model DIR --plan PLAN ... (see the header)');
  process.exit(2);
}
await commands[command]();
pool?.close();
