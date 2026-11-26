#!/usr/bin/env node
/**
 * Draft acceptance against a greedy reference, in JS: for each prompt of the reference, walks the passes a k-token
 * draft would make. At a pass boundary the draft model, holding the accepted context (prompt + reference[:pos]),
 * proposes k greedy tokens; drafts are accepted while they equal the reference; the pass emits accepted + 1 tokens (the
 * last one the target's own). For greedy decoding this gives the accepted lengths a live draft/target pair would get.
 * It is the method of the speculation pre-registration (ouroboros route3 speculative/README.md), ported from its
 * prototype runtime. The draft runs the same JS model code as the requester, with plaintext (dequantized) linears.
 *
 *   node draft-walk.js --model DRAFT_DIR --source-config SRC.json --workloads W.json --reference REF.json --k 4
 *       --out SIM.json [--threads 12] [--only id,id]
 *
 * REF.json: results[].{id, tokens} (or byK['0'].tokens). W.json: [{id, workload, messages}].
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { Tokenizer } from '../../../src/inference/tokenizer.js';
import { QwenRequesterModel, configFrom } from './model.js';
import { fileSource } from './node.js';
import { PlainLinear, planFromManifest } from './plain.js';

const { values: a } = parseArgs({ options: { model: { type: 'string' }, 'source-config': { type: 'string' }, workloads: { type: 'string' },
  reference: { type: 'string' }, k: { type: 'string', default: '4' }, out: { type: 'string' }, threads: { type: 'string', default: '8' }, only: { type: 'string' } } });
const readJson = f => JSON.parse(fs.readFileSync(f, 'utf8'));
const render = messages => messages.map(m => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join('') + '<|im_start|>assistant\n<think>\n\n</think>\n\n';
const argmax = row => { let b = 0; for (let i = 1; i < row.length; i++) if (row[i] > row[b]) b = i; return b; };
const clone = s => ({ pos: s.pos, conv: s.conv.map(x => x && x.slice()), rec: s.rec.map(x => x && x.slice()),
  k: s.k.map(x => x && { data: x.data.slice(0, x.len), len: x.len }), v: s.v.map(x => x && { data: x.data.slice(0, x.len), len: x.len }), snapshot: null });

const source = fileSource(a.model);
const config = configFrom(source.manifest, readJson(a['source-config'] ?? path.join(a.model, 'source-config.json')));
const linear = new PlainLinear(planFromManifest(source.manifest), source, { threads: Number(a.threads) });
const model = new QwenRequesterModel(config, source, linear);
const tokenizer = new Tokenizer();
await tokenizer.initialize(source.manifest, { loadTokenizerJson: async () => readJson(path.join(a.model, 'tokenizer.json')) });
const workloads = new Map(readJson(a.workloads).map(w => [w.id, w]));
const reference = readJson(a.reference);
const K = Number(a.k), only = a.only ? new Set(a.only.split(',')) : null;
const results = [];
const started = performance.now();
for (const r of reference.results) {
  if (only && !only.has(r.id)) continue;
  const ref = r.tokens ?? r.byK['0'].tokens;
  const ids = tokenizer.encode(render(workloads.get(r.id).messages));
  if (r.promptTokens !== undefined && r.promptTokens !== ids.length) throw Error(`${r.id}: ${ids.length} prompt tokens, the reference has ${r.promptTokens}`);
  const state = model.newState();
  // The committed state holds prompt + ref[:pos]; its last logits give the first draft token of the next pass.
  let t0 = performance.now();
  let last = argmax(await model.forward(state, [...ids, ref[0]], { keep: 1 }));
  const prefillMs = performance.now() - t0;
  let pos = 1;
  const passes = [];
  while (pos < ref.length) {
    t0 = performance.now();
    const draft = [last], work = clone(state);
    for (let i = 1; i < K; i++) draft.push(argmax(await model.forward(work, [draft[i - 1]], { keep: 1 })));
    const draftMs = performance.now() - t0;
    let accepted = 0;
    while (accepted < K && pos + accepted < ref.length && draft[accepted] === ref[pos + accepted]) accepted++;
    const emitted = Math.min(accepted + 1, ref.length - pos);
    passes.push({ accepted, emitted, draftMs });
    const feed = ref.slice(pos, pos + emitted);
    pos += emitted;
    if (pos < ref.length) last = argmax(await model.forward(state, feed, { keep: 1 }));
  }
  const histogram = Array.from({ length: K + 1 }, (_, i) => passes.filter(p => p.accepted === i).length);
  results.push({ id: r.id, workload: r.workload, promptTokens: ids.length, referenceTokens: ref.length, prefillMs,
    byK: { [K]: { passes, tokensPerPass: (ref.length - 1) / Math.max(passes.length, 1), acceptedHistogram: histogram } } });
  console.error(JSON.stringify({ id: r.id, k: K, passes: passes.length, tokensPerPass: +((ref.length - 1) / Math.max(passes.length, 1)).toFixed(2),
    meanAccepted: +(passes.reduce((s, p) => s + p.accepted, 0) / Math.max(passes.length, 1)).toFixed(2), elapsedS: +((performance.now() - started) / 1e3).toFixed(0) }));
  fs.writeFileSync(a.out, JSON.stringify({ schema: 'doppler.speculative-draft-sim/v1', runtime: 'javascript (Doppler tools/secret-shared requester model, plaintext dequantized linears, CPU)',
    draftModelId: source.manifest.modelId, referenceModelId: reference.modelId, referenceRuntime: reference.runtime ?? null, k: K, partial: true, results }) + '\n');
}
const all = results.flatMap(r => r.byK[K].passes);
const summary = { passes: all.length, meanAcceptedDrafts: all.reduce((s, p) => s + p.accepted, 0) / all.length,
  tokensPerPass: results.reduce((s, r) => s + r.referenceTokens - 1, 0) / all.length };
fs.writeFileSync(a.out, JSON.stringify({ schema: 'doppler.speculative-draft-sim/v1', runtime: 'javascript (Doppler tools/secret-shared requester model, plaintext dequantized linears, CPU)',
  draftModelId: source.manifest.modelId, referenceModelId: reference.modelId, referenceRuntime: reference.runtime ?? null, k: K, summary, results }) + '\n');
console.error(JSON.stringify({ summary }));
linear.close();
