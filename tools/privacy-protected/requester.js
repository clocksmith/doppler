// The requester: owns the secret key, seals inputs, and decrypts only results it asked for, after checking them. It
// holds no model weights: it knows the alphabet (the tokenizer), the pinned model identity and the profile. It
// answers no evaluator request; a route whose results disagree is closed without decrypting anything.
//
//   node tools/privacy-protected/pp.js requester --profile research-n16-u32 --model-identity sha256:... \
//     --evaluator host:port [--evaluator host:port] | --stage-a host:port --stage-b host:port [--evaluator ...]
//     [--prompt ace] [--generate 4] [--allow-insecure-dev] [--state-dir d] [--key-ttl 24h] [--events] [--report r.json]
//     [--trap-oracle http://host:port --oracle-key <base64url> --circuit sha256:... [--trap-lanes 2]]
//
// Trap lanes (trap verification v2 on the WebGPU route): with a pinned trap oracle, the session carries `trap-lanes`
// extra input streams of fresh secret known-answer traps next to the real one, in an order the evaluator cannot see.
// After every step the requester decrypts the trap lanes first; the real lane is decrypted only if every trap lane is
// within tolerance. A failure closes the route and retires the key epoch.

import { randomBytes, randomInt } from 'node:crypto';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { connect } from 'node:net';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { SecretKey, decrypt } from './ckks/keys.js';
import { KeyWorkers } from './ckks/key-workers.js';
import { encodeCompact } from './ckks/encoder.js';
import { BigFloat } from './ckks/bigfloat.js';
import { ROTATIONS, lookupProfile } from './ckks/params.js';
import { decodeCt } from './ckks/codec.js';
import { ALPHABET, CONTEXT, LOG_SLOTS, PROTOCOL_ID, VOCAB, argmax, replicate, tokenize } from './plan.js';
import { Conn, digest } from './webgpu/wire.js';

export function parseDuration(s) {
  if (/^\d+$/.test(s)) return Number(s);
  let total = 0;
  const re = /(\d+(?:\.\d+)?)(h|m|s)/g;
  let m;
  let consumed = 0;
  while ((m = re.exec(s))) {
    total += Number(m[1]) * { h: 3600, m: 60, s: 1 }[m[2]];
    consumed += m[0].length;
  }
  if (consumed !== s.length || consumed === 0) throw new Error(`cannot parse duration ${s}`);
  return total;
}

function oneHot(tok) {
  const v = new Array(VOCAB).fill(0);
  v[tok] = 1;
  return replicate(v, VOCAB);
}

function dial(addr) {
  const [host, port] = addr.split(/:(?=[^:]*$)/);
  return new Promise((resolve, reject) => {
    const s = connect(Number(port), host, () => resolve(s));
    s.setNoDelay(true);
    s.once('error', reject);
    setTimeout(() => reject(new Error(`could not reach ${addr} within 30 s`)), 30000).unref();
  });
}

// --- key epochs: one secret key and the evaluation keys made from it, reused until they expire ---

function loadOrCreateEpoch(p, stateDir, keyTTL) {
  const now = Math.floor(Date.now() / 1000);
  if (stateDir) {
    try {
      const ep = JSON.parse(readFileSync(join(stateDir, 'key-epoch.json'), 'utf8'));
      const sk = SecretKey.fromBytes(p, readFileSync(join(stateDir, 'key-epoch.sk')));
      if (ep.profileDigest === p.digest && ep.expires > now + 300) return { ep, sk, reused: true };
    } catch { /* no usable epoch: make a new one */ }
  }
  const ep = { epoch: randomBytes(16).toString('hex'), profileDigest: p.digest, created: now, expires: now + Math.floor(keyTTL), keysDigest: '' };
  const sk = SecretKey.generate(p);
  if (stateDir) {
    mkdirSync(stateDir, { recursive: true, mode: 0o700 });
    writeFileSync(join(stateDir, 'key-epoch.sk'), sk.toBytes(), { mode: 0o600 });
    saveEpoch(stateDir, ep);
  }
  return { ep, sk, reused: false };
}

function saveEpoch(dir, ep) {
  if (dir) writeFileSync(join(dir, 'key-epoch.json'), JSON.stringify(ep), { mode: 0o600 });
}

export function retireEpoch(dir) {
  if (!dir) return;
  rmSync(join(dir, 'key-epoch.sk'), { force: true });
  rmSync(join(dir, 'key-epoch.json'), { force: true });
}

// --- the session ---

export async function runRequester(o, rep, start) {
  const p = o.profile;
  if (p.insecure && !o.allowInsecure) throw new Error(`profile ${p.id} is insecure and only allowed with --allow-insecure-dev`);
  if (!o.evaluators.length || !o.modelId.startsWith('sha256:')) throw new Error('an evaluator address and a pinned model identity are required');
  const toks = tokenize(o.prompt);
  if (!toks.length || o.generate < 1 || toks.length + o.generate - 1 > CONTEXT) {
    throw new Error(`prompt of ${toks.length} tokens plus ${o.generate} generated exceeds the ${CONTEXT}-position context`);
  }
  rep.promptTokens = toks;
  const events = o.events ? (e) => process.stdout.write(`${JSON.stringify(e)}\n`) : () => {};
  let stop = false;
  if (o.events) {
    process.stdin.setEncoding('utf8');
    let buf = '';
    process.stdin.on('data', (d) => {
      buf += d;
      for (let i; (i = buf.indexOf('\n')) >= 0; buf = buf.slice(i + 1)) if (buf.slice(0, i).trim() === 'stop') stop = true;
    });
    process.stdin.on('error', () => {});
  }
  // Trap lanes: the real lane's position among the lanes is secret and fresh per session.
  const traps = o.traps;
  const lanes = traps ? traps.traps.length + 1 : 1;
  const realLane = traps ? randomInt(lanes) : 0;
  const laneTraps = traps ? [] : [null];
  if (traps) {
    let t = 0;
    for (let l = 0; l < lanes; l++) laneTraps.push(l === realLane ? null : traps.traps[t++]);
    rep.traps = { source: 'oracle', batch: traps.batchId, lanes, trapLanes: lanes - 1, tolerance: traps.tolerance, checks: 0, maxAbsError: 0 };
  }
  let hopToken = '';
  let roles = [''];
  if (o.split) {
    hopToken = randomBytes(16).toString('hex');
    roles = ['stage-a', 'stage-b'];
    rep.split = { cutLayer: 'ln2', stages: 2, requesterRefreshes: 0,
      refreshReason: 'none needed: the whole block fits the 36-level budget (35 levels used), so no ciphertext is refreshed or bootstrapped between the stages' };
  }
  const peers = [];
  let workers = null;
  try {
    const { ep, sk, reused } = loadOrCreateEpoch(p, o.stateDir, o.keyTTL);
    workers = new KeyWorkers(sk);
    const inputScale = 2 ** p.logScale;
    const seal = (values) => workers.encrypt(encodeCompact(p, values, new BigFloat(128).setFloat64(inputScale), p.maxLevel), p.maxLevel, inputScale);
    rep.keyEpoch = ep.epoch;
    rep.keyEpochExpires = ep.expires;
    rep.keyEpochReusedFromStore = reused;
    const session = randomBytes(16).toString('hex');
    // Admission before upload: each evaluator must state exactly the pinned profile, model and rotations.
    let allHold = reused && ep.keysDigest !== '';
    for (let i = 0; i < o.evaluators.length; i++) {
      const c = new Conn(await dial(o.evaluators[i]), PROTOCOL_ID);
      peers.push(c);
      const fields = { profile: p.id, model: o.modelId, promptTokens: toks.length, generatedTokens: o.generate, keyEpoch: ep.epoch,
        keyEpochExpires: ep.expires, lanes };
      if (o.split && i < 2) Object.assign(fields, { role: roles[i], cutLayer: 'ln2', hopToken });
      await c.send({ kind: 'hello', session, seq: 0, fields });
      const { header: h } = await c.recv('manifest');
      const f = h.fields ?? {};
      if (h.session !== session || f.profile !== p.id || f.profileDigest !== p.digest || f.model !== o.modelId
        || JSON.stringify(f.rotations) !== JSON.stringify(ROTATIONS) || f.inputLevel !== p.maxLevel || f.logSlots !== LOG_SLOTS
        || (f.lanes ?? 1) !== lanes || (o.split && i < 2 && (f.role !== roles[i] || f.cutLayer !== 'ln2'))) {
        throw new Error(`evaluator ${i + 1} does not match the pinned profile, model, lanes or layout; nothing uploaded`);
      }
      allHold = allHold && f.heldKeysDigest === ep.keysDigest;
    }
    const t0 = performance.now();
    let genMs = 0;
    let want = ep.keysDigest;
    rep.evaluationKeyBytesPerEvaluator = 0;
    if (allHold) {
      // Every evaluator still holds this epoch's exact evaluation keys: nothing is uploaded.
      for (const c of peers) await c.send({ kind: 'keys-held', session, seq: 0 });
      rep.evaluationKeysReusedNoUpload = true;
    } else {
      // The secret key never leaves this process. Each evaluation key is generated, sent to every evaluator and
      // dropped, so all evaluators hold identical keys.
      // Keys are generated concurrently on worker threads; each is sent as soon as it and its predecessors are ready.
      const kd = [];
      const g0 = performance.now();
      const keys = workers.evaluationKeys();
      for (;;) {
        const w0 = performance.now();
        const { value: key, done } = await keys.next();
        genMs += performance.now() - w0;
        if (done) break;
        rep.evaluationKeyBytesPerEvaluator += key.length;
        kd.push(digest(key));
        for (const c of peers) await c.send({ kind: 'key', session, seq: 0 }, key);
      }
      want = digest(...kd);
      rep.keygenWallSeconds = (performance.now() - g0) / 1000;
    }
    for (const c of peers) {
      const { header: h } = await c.recv('ready');
      if (h.fields?.keysDigest !== want) throw new Error('evaluator acknowledged different keys');
    }
    if (!allHold) {
      ep.keysDigest = want;
      saveEpoch(o.stateDir, ep);
    }
    // keygenSeconds: time the session waited on key generation (generation overlaps the upload of earlier keys).
    rep.keygenSeconds = genMs / 1000;
    rep.keyUploadSeconds = (performance.now() - t0 - genMs) / 1000;
    events({ event: 'ready', promptTokens: toks.length, keysUploaded: !allHold, keyBytes: rep.evaluationKeyBytesPerEvaluator });

    // step: one request, exactly one result per receiving evaluator, every result checked before any decryption.
    const step = async (seq, kind, laneTokens) => {
      const sr = { seq, kind, inputTokens: laneTokens[0].length, encryptSeconds: 0, uploadBytesPerEvaluator: 0, roundTripSeconds: 0,
        evaluatorReportedComputeSeconds: [], downloadBytesPerEvaluator: 0, verifySeconds: 0, decryptSeconds: 0, outputLevel: null,
        token: -1, logits: [], resultDigest: '' };
      rep.steps.push(sr);
      const te = performance.now();
      const blobs = [];
      for (const lt of laneTokens) for (const t of lt) blobs.push(await seal(oneHot(t)));
      sr.uploadBytesPerEvaluator = blobs.reduce((a, b) => a + b.length, 0);
      sr.encryptSeconds = (performance.now() - te) / 1000;
      const t1 = performance.now();
      let senders = peers;
      let receivers = peers;
      if (o.split) {
        // Inputs go to stage A only; the result comes from stage B only. Whole-model reference evaluators (peers[2:])
        // get the inputs and return results too.
        senders = [peers[0], ...peers.slice(2)];
        receivers = peers.slice(1);
      }
      for (const c of senders) await c.send({ kind, session, seq }, ...blobs);
      const results = [];
      for (const c of receivers) {
        const { header: h, blobs: b } = await c.recv('result');
        if (h.session !== session || h.seq !== seq || b.length !== lanes) throw new Error(`result is not bound to step ${seq} of this session`);
        const f = h.fields ?? {};
        sr.evaluatorReportedComputeSeconds.push(Number(f.computeSeconds) || 0);
        if (o.split && c === peers[1]) {
          sr.stageASeconds = Number(f.stageASeconds) || 0;
          sr.hopBytes = Number(f.hopBytes) || 0;
        }
        sr.outputLevel = f.outputLevel ?? null;
        sr.downloadBytesPerEvaluator = b.reduce((a, x) => a + x.length, 0);
        results.push(b);
      }
      sr.roundTripSeconds = (performance.now() - t1) / 1000;
      const t2 = performance.now();
      // Integrity check: independent evaluators must return byte-identical ciphertexts (evaluation is deterministic
      // for a fixed program and profile). Nothing is decrypted before this passes.
      sr.resultDigest = digest(...results[0]);
      for (const other of results.slice(1)) {
        if (digest(...other) !== sr.resultDigest) throw new Error(`evaluator results differ at step ${seq}; route closed without decrypting`);
      }
      const cts = results[0];
      for (const ct of cts) {
        const shape = decodeCt(p, ct);
        if (shape.deg !== 1) throw new Error('result does not have the pinned shape');
      }
      sr.verifySeconds = (performance.now() - t2) / 1000;
      const t3 = performance.now();
      // Trap lanes first: the real lane is decrypted only if every trap passes.
      if (traps) {
        for (let l = 0; l < lanes; l++) {
          if (l === realLane) continue;
          const trap = laneTraps[l];
          rep.decryptCalls++;
          const logits = decrypt(sk, cts[l]).slice(0, VOCAB);
          const row = trap.expectedLogits[seq - 1];
          const check = o.checkTrap(row, logits, traps.tolerance);
          rep.traps.checks++;
          rep.traps.maxAbsError = Math.max(rep.traps.maxAbsError, check.maxAbsError);
          if (!check.ok) {
            rep.traps.failedAt = { step: seq, maxAbsError: check.maxAbsError, argmaxOk: check.argmaxOk };
            const err = new Error(`trap lane failed at step ${seq} (max error ${check.maxAbsError.toFixed(3)}); route closed, the real lane was not decrypted`);
            err.code = 'TRAP_FAILED';
            throw err;
          }
        }
      }
      rep.decryptCalls++;
      const vals = decrypt(sk, cts[realLane]);
      sr.logits = vals.slice(0, VOCAB);
      sr.token = argmax(sr.logits);
      sr.decryptSeconds = (performance.now() - t3) / 1000;
      return sr.token;
    };
    const emit = (seq, tok) => {
      const sr = rep.steps[rep.steps.length - 1];
      const done = seq === o.generate;
      events({ event: 'token', step: seq - 1, tokenId: tok, delta: ALPHABET[tok], done, stopReason: done ? 'max-tokens' : null,
        inputTokens: sr.inputTokens, wallMs: (sr.encryptSeconds + sr.roundTripSeconds + sr.verifySeconds + sr.decryptSeconds) * 1000,
        roundTripMs: sr.roundTripSeconds * 1000, stageAMs: (sr.stageASeconds ?? 0) * 1000, stageBMs: (sr.evaluatorReportedComputeSeconds[0] ?? 0) * 1000,
        up: sr.uploadBytesPerEvaluator, down: sr.downloadBytesPerEvaluator, hopBytes: sr.hopBytes ?? 0, encryptMs: sr.encryptSeconds * 1000,
        decryptMs: sr.decryptSeconds * 1000 });
    };
    // Fixed public schedule: one prefill message with every prompt token, then generate-1 single-token steps. A
    // supervisor's stop ends the schedule early with bye.
    const laneInputs = (realToks, s) => laneTraps.map((t, l) => (l === realLane ? realToks
      : s === 0 ? t.prompt : [t.continuation[s - 1]]));
    let tok = await step(1, 'prefill', laneInputs(toks, 0));
    rep.timeToFirstTokenSeconds = (performance.now() - start) / 1000;
    const gen = [tok];
    emit(1, tok);
    for (let seq = 2; seq <= o.generate; seq++) {
      if (stop) {
        rep.stoppedAfter = gen.length;
        break;
      }
      tok = await step(seq, 'decode', laneInputs([tok], seq - 1));
      gen.push(tok);
      emit(seq, tok);
    }
    for (const c of peers) await c.send({ kind: 'bye', session, seq: 0 });
    rep.generatedTokens = gen;
    rep.generatedText = gen.map((t) => ALPHABET[t]).join('');
  } catch (e) {
    if (e.code === 'TRAP_FAILED') retireEpoch(o.stateDir);
    throw e;
  } finally {
    rep.sessionBytesPerEvaluator = peers.map((c, i) => ({ evaluator: i + 1, sent: c.sent, received: c.received }));
    for (const c of peers) c.socket.destroy();
    await workers?.close();
  }
}

export async function main(argv = process.argv.slice(2)) {
  const { values } = parseArgs({
    args: argv,
    options: {
      evaluator: { type: 'string', multiple: true, default: [] },
      'stage-a': { type: 'string', default: '' },
      'stage-b': { type: 'string', default: '' },
      events: { type: 'boolean', default: false },
      profile: { type: 'string', default: 'research-n16-u32' },
      'model-identity': { type: 'string', default: '' },
      prompt: { type: 'string', default: 'ace' },
      generate: { type: 'string', default: '4' },
      'allow-insecure-dev': { type: 'boolean', default: false },
      'state-dir': { type: 'string', default: '' },
      'key-ttl': { type: 'string', default: '24h' },
      report: { type: 'string', default: 'requester-report.json' },
      'trap-oracle': { type: 'string', default: '' },
      'oracle-key': { type: 'string', multiple: true, default: [] },
      'trap-lanes': { type: 'string', default: '2' },
      circuit: { type: 'string', default: '' },
    },
  });
  const start = performance.now();
  const p = lookupProfile(values.profile);
  let evaluators = values.evaluator;
  const split = Boolean(values['stage-a'] || values['stage-b']);
  if (split) {
    if (!values['stage-a'] || !values['stage-b']) throw new Error('a split needs --stage-a and --stage-b');
    // Any --evaluator given with a split is a whole-model reference for the dual check.
    evaluators = [values['stage-a'], values['stage-b'], ...evaluators];
  }
  const rep = {
    schema: 'doppler.protected-requester-report/v1', implementation: 'javascript', protocol: PROTOCOL_ID, profile: p.id, profileDigest: p.digest,
    insecureDevelopmentProfile: p.insecure, model: values['model-identity'], integrity: 'none (single evaluator): results decrypted unverified',
    evaluators: evaluators.length, prompt: values.prompt, promptTokens: [], generatedTokens: [], generatedText: '', keygenSeconds: 0,
    evaluationKeyBytesPerEvaluator: 0, keyUploadSeconds: 0, steps: [], decryptCalls: 0, keyEpoch: '', keyEpochExpires: 0,
    keyEpochReusedFromStore: false, evaluationKeysReusedNoUpload: false, timeToFirstTokenSeconds: 0, sessionBytesPerEvaluator: [],
    requesterPeakRSSBytes: 0, totalSeconds: 0, outcome: 'completed',
  };
  if (evaluators.length >= 2 && !split) rep.integrity = 'dual-evaluation: byte-identical results required before decryption';
  if (split) {
    rep.integrity = evaluators.length > 2
      ? "dual-evaluation: the split's result must equal a whole-model evaluator's byte for byte before decryption"
      : 'none (homomorphic split, one evaluator per stage): results decrypted unverified';
  }
  const opts = { profile: p, allowInsecure: values['allow-insecure-dev'], evaluators, modelId: values['model-identity'], prompt: values.prompt,
    generate: Number(values.generate), stateDir: values['state-dir'], keyTTL: parseDuration(values['key-ttl']), split, events: values.events };
  let runErr = null;
  try {
    if (values['trap-oracle']) {
      // The trap batch is fetched and opened before anything reaches an evaluator.
      const { fetchTraps, checkTrapLogits } = await import('./trap-client.js');
      opts.traps = await fetchTraps({ endpoint: values['trap-oracle'], oracleKeys: values['oracle-key'], circuit: values.circuit,
        modelId: values['model-identity'], promptLength: tokenize(values.prompt).length,
        generate: opts.generate, count: Number(values['trap-lanes']) });
      opts.checkTrap = checkTrapLogits;
      rep.integrity = `known-answer traps (v2, oracle): ${opts.traps.traps.length} secret trap lane(s) checked before the real lane is decrypted`;
    }
    await runRequester(opts, rep, start);
  } catch (e) {
    runErr = e;
    rep.outcome = 'refused';
    rep.error = e.message;
    rep.generatedTokens = [];
    rep.generatedText = '';
  }
  rep.requesterPeakRSSBytes = process.resourceUsage().maxRSS * 1024;
  rep.totalSeconds = (performance.now() - start) / 1000;
  writeFileSync(values.report, JSON.stringify(rep, null, 1), { mode: 0o600 });
  if (values.events) {
    const e = runErr ? { event: 'failed', error: runErr.message }
      : { event: 'end', outputTokens: rep.generatedTokens.length, stopReason: rep.stoppedAfter ? 'requester-stop' : 'max-tokens', requesterPeakRss: rep.requesterPeakRSSBytes };
    process.stdout.write(`${JSON.stringify(e)}\n`);
  }
  if (values.events) process.stdin.destroy();
  if (runErr) throw runErr;
}
