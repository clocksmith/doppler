#!/usr/bin/env node
/**
 * Route 3 requester in JavaScript (replaces ../requester.py): Qwen 3.5 with every projection evaluated by a stage's two
 * share operators, on 2-party additive shares in Z_(2^64). Runs in Node here; model.js, cluster.js and ring.js are
 * environment-neutral and take their transport and crypto from the caller, so a browser can host the same requester.
 *
 *   node requester.js generate --model DIR --plan PLAN.json --operators OPS.json --prompts PROMPTS.json --out REPORT.json
 *        [--max-tokens 32] [--forced REPORT.json] [--logits-out PREFIX] [--return-shift 24] [--prg aes256-ctr]
 *        [--pipeline-steps] [--explicit] [--keep-rounds]
 *   node requester.js control --model DIR --plan PLAN.json --prompts PROMPTS.json --out REPORT.json [...]   plaintext linears
 *   node requester.js run --model DIR --plan PLAN.json --operators OPS.json --request REQUEST.json --out REPORT.json [...]
 *        One request for a supervisor: a JSON event per line on stdout (ready, token, end or failed); `stop` on stdin
 *        stops after the token in progress. REQUEST.json: {"messages": [{"role": "user", "content": ...}], "maxTokens": N}.
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
import { PlainLinear } from './plain.js';

const { positionals, values: args } = parseArgs({ allowPositionals: true, options: {
  model: { type: 'string' }, plan: { type: 'string' }, operators: { type: 'string' }, prompts: { type: 'string' },
  request: { type: 'string' }, out: { type: 'string' }, forced: { type: 'string' }, 'logits-out': { type: 'string' },
  'max-tokens': { type: 'string', default: '32' }, 'return-shift': { type: 'string', default: '0' }, prg: { type: 'string', default: 'shake256' },
  'pipeline-steps': { type: 'boolean' }, explicit: { type: 'boolean' }, 'keep-rounds': { type: 'boolean' }, strict: { type: 'boolean' },
  'source-config': { type: 'string' }, 'exit-operators': { type: 'boolean' }, 'prompt-limit': { type: 'string' } } });
const command = positionals[0];
const readJson = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const cpuMs = () => { const u = process.cpuUsage(); return (u.user + u.system) / 1e3; };
const peakRss = () => process.resourceUsage().maxRSS * 1024;
const template = prompt => `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;

async function openModel(linear) {
  const source = fileSource(args.model);
  const config = configFrom(source.manifest, readJson(args['source-config'] ?? path.join(args.model, 'source-config.json')));
  const tokenizer = new Tokenizer();
  await tokenizer.initialize(source.manifest, { loadTokenizerJson: async () => readJson(path.join(args.model, 'tokenizer.json')) });
  return { source, config, tokenizer, model: new QwenRequesterModel(config, source, linear) };
}

async function openCluster(plan) {
  return SharedCluster.open(plan, readJson(args.operators), nodeEnv, {
    seeded: !args.explicit, returnShift: Number(args['return-shift']), prg: args.prg, pipelineSteps: args['pipeline-steps'], strict: args.strict });
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
  const digests = cluster.rounds.map(r => r.shareBSha256), counters = cluster.rounds.map(r => r.counter);
  return { rounds: digests.length, distinctShareBDigests: new Set(digests).size, countersStrictlyIncreasing: counters.every((c, i) => i === 0 || c > counters[i - 1]) };
}

async function cmdGenerate() {
  const plan = readJson(args.plan);
  const t0 = performance.now();
  const cluster = await openCluster(plan);
  const { model, config, tokenizer, source } = await openModel(cluster);
  const loadS = (performance.now() - t0) / 1e3;
  const results = await runPrompts(model, config, tokenizer, cluster);
  const stats = await cluster.close(args['exit-operators'] ? 'exit' : 'close');
  const report = { schema: 'doppler.secret-shared-generation/v1', requester: 'javascript', node: process.version, modelId: plan.modelId, fx: plan.fx, fw: plan.fw,
    seeded: !args.explicit, protocol: { stages: cluster.describe(), pipelineSteps: Boolean(args['pipeline-steps']) }, teacherForced: Boolean(args.forced), loadS,
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
  const request = readJson(args.request);
  if (request.messages?.length !== 1 || request.messages[0].role !== 'user') throw Error('one user message');
  try {
    const t0 = performance.now();
    const plan = readJson(args.plan);
    const cluster = await openCluster(plan);
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
      emit({ event: 'token', step, tokenId: token, delta, done: Boolean(done), stopReason: reason, inputTokens: row.columns,
        ...Object.fromEntries(['wallMs', 'cpuMs', 'rounds', 'up', 'down', 'wireMs', 'operatorMs', 'margin'].map(k => [k, row[k] ?? null])) });
    } });
    const stopped = Boolean(steps.at(-1)?.stoppedAfter);
    const stats = await cluster.close();
    const report = { schema: 'doppler.secret-shared-run/v1', requester: 'javascript', modelId: plan.modelId, fx: plan.fx, fw: plan.fw,
      protocol: { stages: cluster.describe(), pipelineSteps: Boolean(args['pipeline-steps']) },
      requesterHolds: { tensors: source.heldTensors(), bytes: source.heldBytes(), projectionBytes: 0 },
      promptTokens: ids.length, tokens, text, steps, freshness: freshness(cluster), operatorStats: stats, requesterPeakRss: peakRss(), wallS: (performance.now() - t0) / 1e3 };
    fs.writeFileSync(args.out, JSON.stringify(report) + '\n');
    emit({ event: 'end', stopReason: stopped ? 'requester-stop' : tokens.length && config.eos.includes(tokens.at(-1)) ? 'eos' : 'max-tokens',
      outputTokens: tokens.length, freshness: report.freshness, requesterPeakRss: report.requesterPeakRss });
    process.exit(0);
  } catch (error) {
    emit({ event: 'failed', error: `${error?.name}: ${error?.message}`.slice(0, 500) });
    process.exit(1);
  }
}

const commands = { generate: cmdGenerate, control: cmdControl, run: cmdRun };
if (!commands[command] || !args.model || (command !== 'run' && !args.out)) {
  console.error('usage: requester.js generate|control|run --model DIR --plan PLAN ... (see the header)');
  process.exit(2);
}
await commands[command]();
