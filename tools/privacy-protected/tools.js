// Offline tools of the protected route. They load the public weights, so none of them runs inside a requester
// session: gpu-compile (the program compiler), compare (a requester report against the plaintext references after the
// route closed), reference, audit (the approximated circuit against the float64 model) and plain-bench.

import { writeFileSync } from 'node:fs';
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { compileProgram } from './compiler.js';
import { lookupProfile } from './ckks/params.js';
import { evaluatePlaintext, loadTriageModel } from './triage-model.js';
import { CONTEXT as TRIAGE_CONTEXT, GENERATED, MAX_CKKS_LOGIT_ERROR, MAX_QUALITY_DROP, QUALITY_FLOOR, TASK_ID, verifyTaskDigest, WARM_BYTE_TARGET, WORDS } from './triage-task.js';
import {
  ALPHABET, CONTEXT, LOG_SLOTS, PROTOCOL_ID, VOCAB, approxLogits, approximations, argmax, circuitDigest, detokenize, exactLogits, greedy, loadModel, margin, newPlan,
  newTrace, tokenize,
} from './plan.js';

const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

export async function gpuCompile(argv) {
  const { values } = parseArgs({ args: argv, options: {
    profile: { type: 'string', default: 'research-n16-u32' },
    model: { type: 'string' },
    positions: { type: 'string', default: '3' },
    'decode-through': { type: 'string', default: '-1' },
    triage: { type: 'boolean' },
    out: { type: 'string', default: 'gpu-program' },
  } });
  const { json, blob } = compileProgram({ profile: values.profile, modelPath: values.model, triage: values.triage, prompt: Number(values.positions),
    decodeThrough: Number(values['decode-through']), log: (m) => console.error(m) });
  writeFileSync(`${values.out}.json`, json);
  writeFileSync(`${values.out}.bin`, Buffer.from(blob.buffer, blob.byteOffset, blob.byteLength));
}

export function describeTriage(modelPath) {
  const { model, id, circuit } = loadTriageModel(modelPath);
  const p = lookupProfile('research-triage-n14-u32');
  const logQP = [...p.q, ...p.p].reduce((bits, prime) => bits + Math.log2(prime), 0);
  return { task: TASK_ID, taskDigest: verifyTaskDigest(), model: id, circuit, width: model.width,
    protocol: PROTOCOL_ID, scheme: 'ckks-rns', profile: p.id, profileDigest: p.digest, insecureDevelopment: p.insecure,
    tokenizer: { words: WORDS }, vocab: VOCAB, context: TRIAGE_CONTEXT, inputLevel: p.maxLevel, logSlots: LOG_SLOTS, rotations: p.rotations,
    limits: { promptTokens: 2, generatedTokens: GENERATED, qualityFloor: QUALITY_FLOOR, maxQualityDrop: MAX_QUALITY_DROP,
      maxCkksLogitError: MAX_CKKS_LOGIT_ERROR, warmByteTarget: WARM_BYTE_TARGET },
    security: { status: 'guideline-screen-only', ringDimension: p.n, logQP, errorSigma: 3.2, secret: 'uniform-ternary',
      target: '128-bit classical', guidelineLimit: 438, source: 'https://eprint.iacr.org/2024/463', review: 'unreviewed' } };
}

export function triagePlaintext(argv) {
  const { values } = parseArgs({ args: argv, options: { model: { type: 'string' } } });
  const { model } = loadTriageModel(values.model);
  const report = evaluatePlaintext(model, 'holdout');
  console.log(JSON.stringify({ ...report, ...describeTriage(values.model) }));
}

export async function compare(argv) {
  const { values } = parseArgs({ args: argv, options: { model: { type: 'string' }, report: { type: 'string', default: 'requester-report.json' } } });
  const { model, id } = loadModel(values.model);
  const pl = newPlan(model);
  const rep = JSON.parse(readFileSync(values.report, 'utf8'));
  if (rep.model !== id || rep.outcome !== 'completed') throw new Error(`report is for model ${rep.model} with outcome ${JSON.stringify(rep.outcome)}`);
  const n = rep.generatedTokens.length;
  const ex = greedy(rep.promptTokens, n, (t) => exactLogits(model, t));
  const ap = greedy(rep.promptTokens, n, (t) => approxLogits(pl, t));
  let errApprox = 0;
  let errExact = 0;
  let minMargin = Infinity;
  rep.steps.forEach((s, i) => {
    s.logits.forEach((v, j) => {
      errApprox = Math.max(errApprox, Math.abs(v - ap.logits[i][j]));
      errExact = Math.max(errExact, Math.abs(v - ex.logits[i][j]));
    });
    minMargin = Math.min(minMargin, margin(s.logits));
  });
  const out = { model: id, prompt: rep.prompt, encryptedTokens: rep.generatedTokens, approximateReferenceTokens: ap.tokens,
    exactReferenceTokens: ex.tokens, matchesApproximateReference: same(rep.generatedTokens, ap.tokens),
    matchesExactReference: same(rep.generatedTokens, ex.tokens), maxLogitErrorVsApproximate: errApprox, maxLogitErrorVsExact: errExact,
    minEncryptedTop1Margin: minMargin };
  console.log(JSON.stringify(out, null, 1));
  if (!out.matchesApproximateReference || !out.matchesExactReference) throw new Error('encrypted tokens differ from a reference');
}

// reference: the pinned continuation of a prompt (the approximate circuit's greedy tokens and the float64 model's).
export async function reference(argv) {
  const { values } = parseArgs({ args: argv, options: { model: { type: 'string' }, prompt: { type: 'string', default: '' }, generate: { type: 'string', default: '4' } } });
  const { model, id } = loadModel(values.model);
  const pl = newPlan(model);
  const toks = tokenize(values.prompt);
  const g = Number(values.generate);
  if (!toks.length || toks.length + g - 1 > CONTEXT) throw new Error(`prompt must be 1 to ${CONTEXT} alphabet symbols with the generated tokens`);
  const ap = greedy(toks, g, (t) => approxLogits(pl, t));
  const ex = greedy(toks, g, (t) => exactLogits(model, t));
  console.log(JSON.stringify({ model: id, circuit: circuitDigest(pl, id), prompt: values.prompt, approximateText: detokenize(ap.tokens),
    exactText: detokenize(ex.tokens) }));
}

export function auditResult(model, id, pl, steps) {
  const run = (prompts) => {
    const g = { Prompts: 0, TokenMismatch: 0, PromptsWithOutOfRange: 0, NonFinite: 0, MaxLogitError: 0, MinExactMargin: Infinity };
    for (const p of prompts) {
      g.Prompts++;
      const tr = newTrace();
      const ex = greedy(p, steps, (t) => exactLogits(model, t));
      const ap = greedy(p, steps, (t) => approxLogits(pl, t, tr));
      if (!same(ex.tokens, ap.tokens)) g.TokenMismatch++;
      if (Object.keys(tr.outOfRange).length) g.PromptsWithOutOfRange++;
      ex.logits.forEach((row, i) => {
        g.MinExactMargin = Math.min(g.MinExactMargin, margin(row));
        row.forEach((v, j) => {
          const d = Math.abs(v - ap.logits[i][j]);
          if (!Number.isFinite(d)) g.NonFinite++;
          else g.MaxLogitError = Math.max(g.MaxLogitError, d);
        });
      });
    }
    return g;
  };
  const task = [];
  const letters = [];
  for (let a = 0; a < 26; a++) {
    for (let stride = 1; stride <= 3; stride++) task.push([a, (a + stride) % 26, (a + 2 * stride) % 26]);
    for (let b = 0; b < 26; b++) for (let c = 0; c < 26; c++) letters.push([a, b, c]);
  }
  return { model: id, steps, taskPrompts: run(task), allLetterPrompts: run(letters),
    approximations: approximations(pl).map((p) => ({ name: p.name, interval: [p.a, p.b], degree: p.degree })) };
}

export async function audit(argv) {
  const { values } = parseArgs({ args: argv, options: { model: { type: 'string' }, steps: { type: 'string', default: '4' } } });
  const { model, id } = loadModel(values.model);
  console.log(JSON.stringify(auditResult(model, id, newPlan(model), Number(values.steps)), null, 1));
}

// plain-bench: the plaintext forward pass (float64 model and approximated circuit) for one token after a prompt.
export async function plainBench(argv) {
  const { values } = parseArgs({ args: argv, options: { model: { type: 'string' }, prompt: { type: 'string', default: 'hkn' }, reps: { type: 'string', default: '20000' } } });
  const { model } = loadModel(values.model);
  const pl = newPlan(model);
  const toks = tokenize(values.prompt);
  const reps = Number(values.reps);
  let sink = 0;
  let t = performance.now();
  for (let i = 0; i < reps; i++) sink += exactLogits(model, toks)[0];
  const exact = (performance.now() - t) / 1000 / reps;
  t = performance.now();
  for (let i = 0; i < reps; i++) sink += approxLogits(pl, toks)[0];
  const approx = (performance.now() - t) / 1000 / reps;
  console.log(JSON.stringify({ prompt: values.prompt, token: argmax(exactLogits(model, toks)), float64ForwardSeconds: exact,
    approximateCircuitSeconds: approx, reps, checksum: sink }, null, 1));
}

export { ALPHABET };
