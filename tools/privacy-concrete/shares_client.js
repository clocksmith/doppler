/**
 * Requester (client) side of the two-server shares route, in JavaScript. It replaces the Python requester removed at
 * commit 3e485ca93 (shares.py) with the same field, probe, query and check rules, so anchors and receipts stay
 * comparable. It speaks the evaluator pipe protocol (SHQ1/SHR1/SHC1) to any process that serves it: a local
 * shares_evaluator_gpu.js, or `node shares_network.js connect ...` for an enrolled provider over the authenticated,
 * encrypted transport (see buildNetworkCommands). See SHARES.md for the protocol, its claims and its limits.
 *
 * Arithmetic is modulo P = 2^61 - 1. Activation rows are BigUint64Array values below 2^quantizationBits in row-major
 * order; public weights are row-major Int16Array or Int8Array (inputWidth x outputWidth). Each provider only ever
 * sees uniformly random field elements. Research status: production admission remains closed.
 */
import { spawn, spawnSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { lstatSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export const PRIME = (1n << 61n) - 1n;
export const MECHANISM = 'additive-shares-m61-hidden-probe/v1';
const POLICY_BYTES = readFileSync(new URL('shares-policy.json', import.meta.url));
export const POLICY = validatePolicy(JSON.parse(POLICY_BYTES.toString('utf8')));
export const POLICY_ID = createHash('sha256').update(POLICY_BYTES).digest('hex');
export const SHARES_NETWORK_SCRIPT = fileURLToPath(new URL('shares_network.js', import.meta.url));

const REQUEST = Buffer.from('SHQ1', 'latin1');
const RESPONSE = 'SHR1';
const CLOSE = Buffer.from('SHC1', 'latin1');
const REQUEST_HEAD_BYTES = 22;
const RESPONSE_HEAD_BYTES = 21;
const PROBE_DOMAIN = Buffer.from('doppler-shares-probe/v3\0', 'latin1');
const MAX_WIDTH = 1 << 21;
const CLOSE_GRACE_MS = 5000;

export const FAILURE_CODES = Object.freeze(new Set([
  'cancelled', 'deadline-exceeded', 'resource-exhausted', 'contract-rejected', 'integrity-check-failed', 'execution-failed',
]));

/** A bounded public failure code; exception text never crosses this boundary. */
export class ExecutionRejected extends Error {
  constructor(code) {
    if (!FAILURE_CODES.has(code)) throw new Error('Unknown public execution failure code');
    super(code);
    this.code = code;
  }
}

export function validatePolicy(policy) {
  const int = (value, low, high) => Number.isInteger(value) && value >= low && value <= high;
  if (policy?.schema !== 'doppler.two-server-shares-policy/v1' || policy.mechanism !== MECHANISM || policy.fieldBits !== 61
    || !int(policy.checkRounds, 1, 4) || policy.rejection !== 'deferred-until-route-close/v1'
    || !int(policy.maximumRows, 1, 4096) || !int(policy.maximumChunkElements, 1, Number.MAX_SAFE_INTEGER)
    || !int(policy.operationTimeoutSeconds, 1, 3600) || !int(policy.attemptTimeoutSeconds, 1, 86400)) {
    throw new Error('Unsupported two-server shares policy');
  }
  return policy;
}

// Field arithmetic on residues below the prime.
const reduce = (value) => {
  const residue = value % PRIME;
  return residue < 0n ? residue + PRIME : residue;
};
export const add = (left, right) => reduce(left + right);
export const subtract = (left, right) => reduce(left - right);
export const multiply = (left, right) => reduce(left * right);

/** Largest magnitude an honest product can reach: inputs in [0, 2^q), weights in [-2^(q-1), 2^(q-1)). */
export function productBound(inputWidth, quantizationBits) {
  const bound = BigInt(inputWidth) * (1n << BigInt(quantizationBits)) * (1n << BigInt(quantizationBits - 1));
  if (bound >= 1n << 59n || inputWidth >= MAX_WIDTH) throw new Error('Projection exceeds the share field');
  return bound;
}

/** One evaluator that changes an output value must guess a field element per round: at most 2^-this. */
export const detectionLog2 = (rounds) => 60 * rounds;
export const COMMON_DEVIATION_LOG2 = 60;

/** Exactly uniform field elements: a draw equal to the prime (or zero, when excluded) is drawn again. */
export function randomField(count, rng = randomBytes, nonzero = false) {
  const out = new BigUint64Array(count);
  for (let filled = 0; filled < count;) {
    const bytes = rng(8 * (count - filled));
    for (let offset = 0; offset + 8 <= bytes.length && filled < count; offset += 8) {
      const value = bytes.readBigUInt64LE(offset) & PRIME;
      if (value === PRIME || (nonzero && value === 0n)) continue;
      out[filled++] = value;
    }
  }
  return out;
}

/** The requester's secret probe row for a projection. Providers never see it or anything that reveals it. */
export function probe(seed, name, width) {
  if (!Buffer.isBuffer(seed) || seed.length !== 32 || !Number.isInteger(width) || width <= 0 || width >= MAX_WIDTH) {
    throw new Error('A 32-byte secret probe seed and an explicit width are required');
  }
  const encoded = Buffer.from(name, 'utf8');
  const header = Buffer.alloc(8);
  header.writeUInt32LE(width, 0);
  header.writeUInt32LE(encoded.length, 4);
  const stream = createHash('shake256', { outputLength: 8 * (width + 16) })
    .update(Buffer.concat([PROBE_DOMAIN, seed, header.subarray(0, 4), header.subarray(4), encoded]))
    .digest();
  const values = new BigUint64Array(width);
  let filled = 0;
  for (let offset = 0; offset < stream.length && filled < width; offset += 8) {
    const value = stream.readBigUInt64LE(offset) & PRIME;
    if (value !== PRIME) values[filled++] = value;
  }
  if (filled !== width) throw new Error('Probe derivation failed');
  return values;
}

/**
 * rows (count x inputWidth field elements) times weights (inputWidth x outputWidth small signed integers) modulo the
 * prime. Each element is split into four 16-bit limbs; a limb times a 16-bit weight summed over fewer than 2^21
 * inputs stays below 2^53, so the double accumulators are exact.
 */
export function fieldProduct(rows, count, weights, inputWidth, outputWidth) {
  if (!(rows instanceof BigUint64Array) || rows.length !== count * inputWidth || inputWidth >= MAX_WIDTH
    || !(weights instanceof Int16Array || weights instanceof Int8Array) || weights.length !== inputWidth * outputWidth) {
    throw new Error('Explicit share geometry required');
  }
  const out = new BigUint64Array(count * outputWidth);
  const limbs = [0, 1, 2, 3].map(() => new Float64Array(outputWidth));
  for (let r = 0; r < count; r += 1) {
    for (const limb of limbs) limb.fill(0);
    for (let i = 0; i < inputWidth; i += 1) {
      const value = rows[r * inputWidth + i];
      const lo = Number(value & 0xffffffffn), hi = Number(value >> 32n);
      const parts = [lo & 0xffff, lo >>> 16, hi & 0xffff, hi >>> 16];
      const base = i * outputWidth;
      for (let k = 0; k < 4; k += 1) {
        const part = parts[k];
        if (part === 0) continue;
        const limb = limbs[k];
        for (let j = 0; j < outputWidth; j += 1) limb[j] += part * weights[base + j];
      }
    }
    for (let j = 0; j < outputWidth; j += 1) {
      let total = 0n;
      for (let k = 0; k < 4; k += 1) total += reduce(BigInt(limbs[k][j])) << BigInt(16 * k);
      out[r * outputWidth + j] = reduce(total);
    }
  }
  return out;
}

/** The probe's exact product with the true matrix. Computed once by the requester, or by someone it trusts. */
export function buildAnchor(seed, name, weights, inputWidth, outputWidth) {
  return fieldProduct(probe(seed, name, inputWidth), 1, weights, inputWidth, outputWidth);
}

/**
 * Rows for providers A and B, and the secrets the requester keeps to check their answers.
 * A receives x1, then per round alpha*x1 + delta*probe + t, then per round u.
 * B receives x2, then per round beta*x2 + epsilon*probe + u, then per round t.
 */
export function buildQueries(inputs, count, width, probeRow, rounds, rng = randomBytes) {
  const first = randomField(count * width, rng);
  const second = new BigUint64Array(count * width);
  for (let index = 0; index < second.length; index += 1) second[index] = subtract(inputs[index], first[index]);
  const scalars = () => randomField(rounds * count, rng);
  const alpha = scalars(), beta = scalars();
  const delta = randomField(rounds * count, rng, true), epsilon = randomField(rounds * count, rng, true);
  const t = randomField(rounds * count * width, rng), u = randomField(rounds * count * width, rng);
  const blocks = 1 + 2 * rounds;
  const a = new BigUint64Array(blocks * count * width);
  const b = new BigUint64Array(blocks * count * width);
  a.set(first, 0);
  b.set(second, 0);
  for (let k = 0; k < rounds; k += 1) {
    const checkAt = (1 + k) * count * width, maskAt = (1 + rounds + k) * count * width;
    for (let r = 0; r < count; r += 1) {
      const s = k * count + r;
      for (let i = 0; i < width; i += 1) {
        const at = r * width + i, maskIndex = k * count * width + at;
        a[checkAt + at] = add(add(multiply(alpha[s], first[at]), multiply(delta[s], probeRow[i])), t[maskIndex]);
        b[checkAt + at] = add(add(multiply(beta[s], second[at]), multiply(epsilon[s], probeRow[i])), u[maskIndex]);
      }
    }
    a.set(u.subarray(k * count * width, (k + 1) * count * width), maskAt);
    b.set(t.subarray(k * count * width, (k + 1) * count * width), maskAt);
  }
  return { a, b, secrets: { alpha, beta, delta, epsilon, rows: count, rounds } };
}

/** Exact signed product (BigInt64Array) and whether every check passed. Nothing here depends on which check failed. */
export function combine(answerA, answerB, secrets, anchor, bound, outputWidth) {
  const { rows, rounds } = secrets;
  const expected = (1 + 2 * rounds) * rows * outputWidth;
  if (!(answerA instanceof BigUint64Array) || !(answerB instanceof BigUint64Array)
    || answerA.length !== expected || answerB.length !== expected
    || answerA.some((value) => value >= PRIME) || answerB.some((value) => value >= PRIME)) {
    return { product: null, passed: false };
  }
  const at = (block, r, j) => (block * rows + r) * outputWidth + j;
  let passed = true;
  for (let k = 0; k < rounds; k += 1) {
    for (let r = 0; r < rows; r += 1) {
      const s = k * rows + r;
      for (let j = 0; j < outputWidth; j += 1) {
        const left = subtract(subtract(answerA[at(1 + k, r, j)], multiply(secrets.alpha[s], answerA[at(0, r, j)])), answerB[at(1 + rounds + k, r, j)]);
        const right = subtract(subtract(answerB[at(1 + k, r, j)], multiply(secrets.beta[s], answerB[at(0, r, j)])), answerA[at(1 + rounds + k, r, j)]);
        passed = (left === multiply(secrets.delta[s], anchor[j])) && passed;
        passed = (right === multiply(secrets.epsilon[s], anchor[j])) && passed;
      }
    }
  }
  const product = new BigInt64Array(rows * outputWidth);
  for (let index = 0; index < product.length; index += 1) {
    const total = add(answerA[index], answerB[index]);
    const signed = total > PRIME / 2n ? total - PRIME : total;
    // A sanity bound, not the integrity mechanism: a wrong value only reaches it after passing the field checks.
    if (!(signed > -bound && signed < bound)) passed = false;
    product[index] = signed;
  }
  return { product, passed };
}

/** One provider reached through a child process's pipes: a local evaluator or a shares_network.js connect command. */
export class ProcessEvaluator {
  constructor(command, label, { timeoutMs = POLICY.operationTimeoutSeconds * 1000 } = {}) {
    if (!Array.isArray(command) || command.length === 0 || !command.every((part) => typeof part === 'string' && part)) {
      throw new Error('A non-empty string-list evaluator command is required');
    }
    Object.assign(this, { label, timeoutMs, sequence: 0n, sent: 0, received: 0, failed: false, deadline: null });
    // Its own process group, so a deadline also ends any helper that inherited the pipes.
    this.process = spawn(command[0], command.slice(1), { stdio: ['pipe', 'pipe', 'ignore'], detached: process.platform !== 'win32' });
    this.exited = new Promise((resolve) => this.process.once('close', resolve));
    this.process.once('error', () => { this.failed = true; });
    this.process.stdin.on('error', () => { this.failed = true; });
    this.chunks = [];
    this.buffered = 0;
    this.ended = false;
    this.wake = null;
    this.process.stdout.on('data', (chunk) => { this.chunks.push(chunk); this.buffered += chunk.length; this.wake?.(); });
    this.process.stdout.on('close', () => { this.ended = true; this.wake?.(); });
  }

  kill() {
    try { process.kill(-this.process.pid, 'SIGKILL'); } catch {}
    try { this.process.kill('SIGKILL'); } catch {}
  }

  /** One operation, bounded by the operation timeout and the attempt deadline. */
  async bounded(operation) {
    const limit = Math.min(this.timeoutMs, this.deadline === null ? Infinity : this.deadline - Date.now());
    let timer = null;
    const expiry = new Promise((_, reject) => {
      timer = setTimeout(() => {
        this.failed = true;
        this.kill();
        reject(new ExecutionRejected(this.deadline !== null && Date.now() >= this.deadline ? 'deadline-exceeded' : 'execution-failed'));
      }, Math.max(0, limit));
    });
    try {
      return await Promise.race([operation(), expiry]);
    } catch (error) {
      this.failed = true;
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  async read(length) {
    while (this.buffered < length) {
      if (this.ended) throw new Error('Evaluator closed its stream');
      await new Promise((resolve) => { this.wake = resolve; });
      this.wake = null;
    }
    const all = this.chunks.length === 1 ? this.chunks[0] : Buffer.concat(this.chunks);
    this.chunks = all.length > length ? [all.subarray(length)] : [];
    this.buffered -= length;
    this.received += length;
    return all.subarray(0, length);
  }

  write(buffer) {
    return new Promise((resolve, reject) => {
      if (this.process.stdin.destroyed) return reject(new Error('Evaluator input closed'));
      this.process.stdin.write(buffer, (error) => (error ? reject(error) : resolve()));
    });
  }

  async send(name, rows, count, width) {
    const encoded = Buffer.from(name, 'utf8');
    this.sequence += 1n;
    const head = Buffer.alloc(REQUEST_HEAD_BYTES);
    REQUEST.copy(head, 0);
    head.writeBigUInt64LE(this.sequence, 4);
    head.writeUInt16LE(encoded.length, 12);
    head.writeUInt32LE(count, 14);
    head.writeUInt32LE(width, 18);
    const payload = Buffer.from(rows.buffer, rows.byteOffset, rows.byteLength);
    await this.bounded(() => this.write(Buffer.concat([head, encoded, payload])));
    this.sent += head.length + encoded.length + payload.length;
    return this.sequence;
  }

  /** Answer to the request with sequence `expected` (default: the one sent last), as count x width field elements. */
  async receive(count, width, expected = this.sequence) {
    return this.bounded(async () => {
      const head = await this.read(RESPONSE_HEAD_BYTES);
      if (head.toString('latin1', 0, 4) !== RESPONSE || head.readBigUInt64LE(4) !== expected || head.readUInt8(12) !== 0
        || head.readUInt32LE(13) !== count || head.readUInt32LE(17) !== width) {
        throw new Error('Evaluator answered out of contract');
      }
      const bytes = await this.read(8 * count * width);
      const answer = new BigUint64Array(count * width);
      new Uint8Array(answer.buffer).set(bytes);
      return answer;
    });
  }

  /** After a failed exchange the provider is ended, not asked to stop. A polite close is itself bounded. */
  async close() {
    if (!this.failed && this.process.exitCode === null && !this.process.stdin.destroyed) {
      try { this.process.stdin.end(CLOSE); } catch { this.failed = true; }
    }
    if (this.failed) this.kill();
    const timer = setTimeout(() => this.kill(), CLOSE_GRACE_MS);
    await this.exited;
    clearTimeout(timer);
    return this.process.exitCode;
  }
}

/**
 * Requester side of one attempt under the installed policy. Holds no weight matrix: its secret probe seed, one anchor
 * row per projection and its own randomness. A failed check is recorded and answered with zeros; the caller must
 * call requireIntegrity() after close() and release nothing if it throws. `geometry`, when given, maps each
 * projection name to [inputWidth, outputWidth].
 */
export class ShareRoute {
  constructor(evaluators, probeSeed, anchors, { geometry = null, rng = randomBytes, policy = POLICY } = {}) {
    if (!Array.isArray(evaluators) || evaluators.length !== 2) throw new Error('Exactly two evaluators required');
    [this.a, this.b] = evaluators;
    Object.assign(this, { probeSeed, anchors, geometry, rng, policy });
    Object.assign(this, { began: null, calls: 0, rows: 0, failure: null, closed: false, clientMs: 0, remoteMs: 0 });
  }

  checkActive() {
    if (this.closed) throw new ExecutionRejected('contract-rejected');
    if (this.began !== null && Date.now() - this.began > this.policy.attemptTimeoutSeconds * 1000) {
      this.closed = true;
      throw new ExecutionRejected(this.failure !== null ? 'integrity-check-failed' : 'deadline-exceeded');
    }
  }

  /** Exact inputs @ W (BigInt64Array, count x outputWidth) for the named projection, or zeros after a failed check. */
  async linear(name, inputs, count, outputWidth, quantizationBits) {
    this.checkActive();
    const width = inputs.length / count;
    const anchor = this.anchors.get(name);
    const limit = 1n << BigInt(quantizationBits);
    const expectedGeometry = this.geometry?.get(name);
    if (!(inputs instanceof BigUint64Array) || !Number.isInteger(count) || count < 1 || count > this.policy.maximumRows
      || !Number.isInteger(width) || !anchor || anchor.length !== outputWidth
      || (this.geometry !== null && (!expectedGeometry || expectedGeometry[0] !== width || expectedGeometry[1] !== outputWidth))
      || inputs.some((value) => value >= limit)) {
      throw new Error('Projection is outside the admitted share route');
    }
    this.began ??= Date.now();
    const deadline = this.began + this.policy.attemptTimeoutSeconds * 1000;
    this.a.deadline = deadline;
    this.b.deadline = deadline;
    const rounds = this.policy.checkRounds;
    const bound = productBound(width, quantizationBits);
    const started = performance.now();
    const { a, b, secrets } = buildQueries(inputs, count, width, probe(this.probeSeed, name, width), rounds, this.rng);
    const blockRows = (1 + 2 * rounds) * count;
    const built = performance.now();
    let answers;
    try {
      // Both providers work at once; each side is one request and its answer.
      answers = await Promise.all([[this.a, a], [this.b, b]].map(async ([evaluator, rows]) => {
        const sequence = await evaluator.send(name, rows, blockRows, width);
        return evaluator.receive(blockRows, outputWidth, sequence);
      }));
    } catch (error) {
      // A provider that was already caught cannot turn that into a transport error by leaving.
      const reason = error instanceof ExecutionRejected ? error.code
        : Date.now() >= deadline ? 'deadline-exceeded' : 'execution-failed';
      this.a.failed = true;
      this.b.failed = true;
      await this.close();
      throw new ExecutionRejected(this.failure !== null ? 'integrity-check-failed' : reason);
    }
    const answered = performance.now();
    const { product, passed } = combine(answers[0], answers[1], secrets, anchor, bound, outputWidth);
    this.calls += 1;
    this.rows += count;
    this.clientMs += (built - started) + (performance.now() - answered);
    this.remoteMs += answered - built;
    if (!passed) {
      this.failure ??= 'check';
      return new BigInt64Array(count * outputWidth);
    }
    return product;
  }

  integrityFailed() { return this.failure !== null; }

  requireIntegrity() {
    if (this.failure !== null) throw new ExecutionRejected('integrity-check-failed');
  }

  async close() {
    this.closed = true;
    return Promise.all([this.a.close(), this.b.close()]);
  }

  report() {
    return {
      integrity: MECHANISM,
      integrityPolicyId: POLICY_ID,
      checkRounds: this.policy.checkRounds,
      sharedProjectionCalls: this.calls,
      sharedRows: this.rows,
      singleEvaluatorDetectionLog2: detectionLog2(this.policy.checkRounds),
      commonDeviationDetectionLog2: COMMON_DEVIATION_LOG2,
      clientShareMs: this.clientMs,
      evaluatorRoundTripMs: this.remoteMs,
      sentBytes: [this.a.sent, this.b.sent],
      receivedBytes: [this.a.received, this.b.received],
    };
  }
}

/**
 * Commands that reach two enrolled providers through shares_network.js connect, after verify-enrollment proves they
 * are distinct enrolled computers with different owners. providerEndpoints: ['<computerId>@<host>:<port>', ...].
 */
export function buildNetworkCommands({
  enrollmentFile, providerEndpoints, planId, modelId, manifestSha256, requesterOwner = null, node = process.execPath,
}) {
  if (!Array.isArray(providerEndpoints) || providerEndpoints.length !== 2) {
    throw new Error('Exactly two provider endpoints (<computerId>@<host>:<port>) required');
  }
  const stat = lstatSync(enrollmentFile);
  if (stat.isSymbolicLink() || !stat.isFile()) throw new Error('Regular non-symlink enrollment file required');
  const enrollmentSha256 = createHash('sha256').update(readFileSync(enrollmentFile)).digest('hex');
  const parsed = providerEndpoints.map((item) => {
    const at = typeof item === 'string' ? item.indexOf('@') : -1;
    const computerId = at > 0 ? item.slice(0, at) : '';
    const endpoint = at > 0 ? item.slice(at + 1).trim() : '';
    if (!computerId || computerId.startsWith('--') || !endpoint.includes(':') || endpoint.startsWith('--')) {
      throw new Error(`Invalid provider endpoint ${item}`);
    }
    return { computerId, endpoint };
  });
  if (parsed[0].computerId === parsed[1].computerId) throw new Error('Duplicate provider computerId');
  if (parsed[0].endpoint.toLowerCase() === parsed[1].endpoint.toLowerCase()) throw new Error('Duplicate provider endpoint');
  if (requesterOwner !== null && (!requesterOwner.trim() || requesterOwner.startsWith('--'))) {
    throw new Error('Invalid requesterOwner');
  }
  const owner = requesterOwner === null ? [] : ['--requester-owner', requesterOwner];
  const checked = spawnSync(node, [
    SHARES_NETWORK_SCRIPT, 'verify-enrollment', '--enrollment', enrollmentFile, '--enrollment-sha256', enrollmentSha256,
    '--provider-a', parsed[0].computerId, '--provider-b', parsed[1].computerId, ...owner,
  ], { encoding: 'utf8', timeout: 15_000 });
  if (checked.status !== 0) throw new Error(checked.stderr.trim() || 'Provider enrollment verification failed');
  const verification = JSON.parse(checked.stdout);
  const commands = parsed.map(({ computerId, endpoint }, index) => [
    node, SHARES_NETWORK_SCRIPT, 'connect', '--endpoint', endpoint, '--enrollment', enrollmentFile,
    '--enrollment-sha256', enrollmentSha256, '--computer-id', computerId, '--peer-computer-id', parsed[1 - index].computerId,
    '--role', index === 0 ? 'a' : 'b', '--plan-id', planId, '--model-id', modelId,
    '--policy-id', POLICY_ID, '--manifest-sha256', manifestSha256, ...owner,
  ]);
  return { commands, verification };
}
