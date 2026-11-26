/**
 * Speculative verification for the secret-shared requester: verify k drafted tokens in one protected pass and keep
 * the prefix the model would itself have produced greedily, plus the model's own next token.
 *
 * A pass is one `model.forward` over the last accepted token and k draft slots, so it costs the same share rounds as
 * a one-token step (one per projection group), with k + 1 columns in each. Every pass has exactly k + 1 columns: slots
 * the draft cannot fill are padded and computed like any other column, so the operators cannot see how many guesses
 * matched. The requester holds the KV and DeltaNet state; `model.commit(state, accepted + 1)` keeps the accepted
 * positions and rolls back the rest without a round, and the bonus token is the next pass's column 0.
 * What still leaks: passes per answer and the answer length, hence the answer's mean acceptance length.
 *
 * Environment-neutral; the caller supplies the model (model.js), its state and, optionally, the share cluster for
 * cost accounting.
 */

/** Prompt lookup: what followed the most recent earlier occurrence of the context's last n-gram (n = maxNgram..1). */
export function promptLookupDraft(context, k, { maxNgram = 3 } = {}) {
  for (let n = Math.min(maxNgram, context.length - 1); n >= 1; n--) {
    for (let i = context.length - n - 1; i >= 0; i--) {
      let match = true;
      for (let j = 0; j < n && match; j++) match = context[i + j] === context[context.length - n + j];
      if (!match) continue;
      const found = context.slice(i + n, i + n + k);
      if (found.length) return found;
    }
  }
  return [];
}

/** Fixed width: unfilled slots repeat the draft's last token, or the context's last token. */
export function padDraft(draft, k, context) {
  const out = draft.slice(0, k), padded = k - out.length;
  const filler = out.length ? out[out.length - 1] : context[context.length - 1];
  for (let i = 0; i < padded; i++) out.push(filler);
  return { draft: out, padded };
}

/** Drafts are kept while each equals the model's token after the previous column; the model's next token follows. */
export function acceptGreedy(draft, argmax) {
  if (argmax.length !== draft.length + 1) throw RangeError('one model token per column');
  let accepted = 0;
  while (accepted < draft.length && draft[accepted] === argmax[accepted]) accepted++;
  return { accepted, tokens: [...draft.slice(0, accepted), argmax[accepted]] };
}

/** Greedy token of each row of [rows * vocab] logits (first maximum, as the sequential path). */
export function argmaxRows(logits, rows, vocab) {
  const out = [];
  for (let r = 0; r < rows; r++) {
    let top = r * vocab;
    for (let i = r * vocab + 1; i < (r + 1) * vocab; i++) if (logits[i] > logits[top]) top = i;
    out.push(top - r * vocab);
  }
  return out;
}

/** A deep copy of a requester state (model.newState() layout), so several k values decode from one prefill. */
export function cloneState(state) {
  if (state.snapshot) throw Error('clone a committed state');
  const kv = c => c && { data: c.data.slice(0, c.len), len: c.len };
  return { pos: state.pos, conv: state.conv.map(a => a && a.slice()), rec: state.rec.map(a => a && a.slice()),
    k: state.k.map(kv), v: state.v.map(kv), snapshot: null };
}

/**
 * Decodes from a prefilled state whose prefill produced `first`. k = 0 is the sequential protocol (one column per
 * pass, no snapshot). Returns { tokens, passes } with one cost record per pass.
 */
export async function speculativeDecode(model, state, { promptIds, first, k, eos, maxTokens, cluster = null, draft = promptLookupDraft,
  cpuMs = () => 0, endStep = true } = {}) {
  const vocab = model.c.vocab, stop = new Set(eos);
  const tokens = [first], passes = [];
  const base = cluster ? (cluster.step === null ? 0 : cluster.step + 1) : 0;
  while (!stop.has(tokens[tokens.length - 1]) && tokens.length < maxTokens) {
    const index = passes.length;
    if (cluster) cluster.step = base + index;
    const before = cluster ? cluster.rounds.length : 0, bytes0 = cluster?.bytes();
    const t0 = performance.now(), c0 = cpuMs();
    const context = [...promptIds, ...tokens];
    const { draft: drafted, padded } = k === 0 ? { draft: [], padded: 0 } : padDraft(await draft(context, k), k, context);
    const draftMs = performance.now() - t0;
    const columns = [tokens[tokens.length - 1], ...drafted];
    const logits = await model.forward(state, columns, { keep: columns.length, speculative: k > 0 });
    const argmax = argmaxRows(logits, columns.length, vocab);
    const { accepted, tokens: produced } = acceptGreedy(drafted, argmax);
    if (k > 0) model.commit(state, accepted + 1);
    let emitted = 0;
    for (const token of produced) {
      if (stop.has(tokens[tokens.length - 1]) || tokens.length >= maxTokens) break;
      tokens.push(token);
      emitted++;
    }
    const row = { pass: index, columns: columns.length, accepted, emitted, padded, draftMs, wallMs: performance.now() - t0, cpuMs: cpuMs() - c0 };
    if (cluster) {
      const rounds = cluster.rounds.slice(before), bytes1 = cluster.bytes();
      Object.assign(row, { rounds: rounds.length, up: bytes1.sent - bytes0.sent, down: bytes1.received - bytes0.received,
        wireMs: rounds.reduce((s, r) => s + r.wireMs, 0), operatorMs: rounds.reduce((s, r) => s + Math.max(...r.operatorMs), 0),
        encodeMs: rounds.reduce((s, r) => s + r.encodeMs, 0), decodeMs: rounds.reduce((s, r) => s + r.decodeMs, 0) });
      if (endStep) {
        const e0 = performance.now();
        await cluster.endStep(base + index);
        row.endStepMs = performance.now() - e0;
        row.wallMs += row.endStepMs;
      }
    }
    passes.push(row);
  }
  return { tokens, passes };
}

/** Per-answer cost per accepted (decode) token; the prefill's own token is not counted. */
export function summarize(tokens, passes) {
  const decoded = tokens.length - 1;
  if (!passes.length || decoded <= 0) return null;
  const sum = key => passes.reduce((s, p) => s + (p[key] ?? 0), 0);
  const histogram = new Array(passes[0].columns).fill(0);
  for (const p of passes) histogram[p.accepted]++;
  return { passes: passes.length, decodedTokens: decoded, wallS: sum('wallMs') / 1e3, draftS: sum('draftMs') / 1e3,
    sPerToken: sum('wallMs') / 1e3 / decoded, roundsPerToken: sum('rounds') / decoded, bytesPerToken: (sum('up') + sum('down')) / decoded,
    upPerToken: sum('up') / decoded, downPerToken: sum('down') / decoded, acceptedHistogram: histogram,
    meanAccepted: passes.reduce((s, p) => s + p.accepted, 0) / passes.length, paddedSlots: sum('padded') };
}
