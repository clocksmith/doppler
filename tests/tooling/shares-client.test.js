import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import {
  ExecutionRejected,
  POLICY,
  PRIME,
  ProcessEvaluator,
  SHARES_NETWORK_SCRIPT,
  ShareRoute,
  buildAnchor,
  buildNetworkCommands,
  fieldProduct,
  probe,
} from '../../tools/privacy-concrete/shares_client.js';

const EVALUATOR = resolve('tests/fixtures/shares/cpu-evaluator.js');
const NAME = 'model.layers.0.mlp.up_proj';
const INPUT_WIDTH = 24;
const OUTPUT_WIDTH = 7;
const ROWS = 3;
const BITS = 12;

// Field product against a plain BigInt reference, including negative weights and full 61-bit residues.
{
  const weights = Int16Array.from({ length: 4 * 3 }, (_, index) => (index * 7919) % 65536 - 32768);
  const rows = BigUint64Array.from([PRIME - 1n, 0n, 1n, (1n << 60n) + 12345n, 5n, PRIME - 2n, 77n, 1n]);
  const product = fieldProduct(rows, 2, weights, 4, 3);
  for (let r = 0; r < 2; r += 1) {
    for (let j = 0; j < 3; j += 1) {
      let expected = 0n;
      for (let i = 0; i < 4; i += 1) expected += rows[r * 4 + i] * BigInt(weights[i * 3 + j]);
      expected = ((expected % PRIME) + PRIME) % PRIME;
      assert.equal(product[r * 3 + j], expected);
    }
  }
}

// The probe is deterministic per seed, name and width, and stays below the prime.
{
  const seed = Buffer.alloc(32, 9);
  const first = probe(seed, NAME, 5);
  assert.deepEqual(first, probe(seed, NAME, 5));
  assert.notDeepEqual(first, probe(seed, `${NAME}x`, 5));
  assert.ok(first.every((value) => value < PRIME));
  assert.throws(() => probe(Buffer.alloc(16), NAME, 5), /32-byte/);
}

const directory = mkdtempSync(join(tmpdir(), 'doppler-shares-client-'));
const weights = Int16Array.from({ length: INPUT_WIDTH * OUTPUT_WIDTH }, () => Math.floor(Math.random() * 4096) - 2048);
const weightBytes = Buffer.from(weights.buffer);
writeFileSync(join(directory, 'up.bin'), weightBytes);
const manifestBytes = Buffer.from(JSON.stringify({
  schema: 'doppler.two-server-shares-manifest/v1',
  projections: [{
    name: NAME,
    file: 'up.bin',
    dtype: '<i2',
    weightId: createHash('sha256').update(weightBytes).digest('hex'),
    inputWidth: INPUT_WIDTH,
    outputWidth: OUTPUT_WIDTH,
  }],
}));
writeFileSync(join(directory, 'shares-manifest.json'), manifestBytes);

const seed = randomBytes(32);
const anchors = new Map([[NAME, buildAnchor(seed, NAME, weights, INPUT_WIDTH, OUTPUT_WIDTH)]]);
const geometry = new Map([[NAME, [INPUT_WIDTH, OUTPUT_WIDTH]]]);
const inputs = BigUint64Array.from({ length: ROWS * INPUT_WIDTH }, () => BigInt(Math.floor(Math.random() * (1 << BITS))));
const expected = new BigInt64Array(ROWS * OUTPUT_WIDTH);
for (let r = 0; r < ROWS; r += 1) {
  for (let j = 0; j < OUTPUT_WIDTH; j += 1) {
    let sum = 0n;
    for (let i = 0; i < INPUT_WIDTH; i += 1) sum += inputs[r * INPUT_WIDTH + i] * BigInt(weights[i * OUTPUT_WIDTH + j]);
    expected[r * OUTPUT_WIDTH + j] = sum;
  }
}
const evaluatorCommand = (...extra) => [process.execPath, EVALUATOR, '--cache-dir', directory, ...extra];

async function runRoute(commands) {
  const route = new ShareRoute(commands.map((command, index) => new ProcessEvaluator(command, 'ab'[index])), seed, anchors, { geometry });
  try {
    const first = await route.linear(NAME, inputs, ROWS, OUTPUT_WIDTH, BITS);
    const second = await route.linear(NAME, inputs, ROWS, OUTPUT_WIDTH, BITS);
    return { route, first, second };
  } finally {
    await route.close();
  }
}

try {
  // Two honest local providers: the exact product, twice over one route, and a passing check.
  {
    const { route, first, second } = await runRoute([evaluatorCommand(), evaluatorCommand()]);
    assert.deepEqual(first, expected);
    assert.deepEqual(second, expected);
    route.requireIntegrity();
    const report = route.report();
    assert.equal(report.sharedProjectionCalls, 2);
    assert.equal(report.sharedRows, 2 * ROWS);
    assert.equal(report.checkRounds, POLICY.checkRounds);
    const rowsPerSide = (1 + 2 * POLICY.checkRounds) * ROWS;
    assert.equal(report.receivedBytes[0], 2 * (21 + 8 * rowsPerSide * OUTPUT_WIDTH));
  }

  // A provider that changes one output element of its real block is caught; the route releases only zeros.
  for (const fault of [0, (1 + POLICY.checkRounds) * ROWS * OUTPUT_WIDTH]) {
    const { route, first } = await runRoute([evaluatorCommand(), evaluatorCommand('--fault', String(fault))]);
    assert.ok(first.every((value) => value === 0n));
    assert.throws(() => route.requireIntegrity(), (error) => error instanceof ExecutionRejected && error.code === 'integrity-check-failed');
  }

  // Out-of-range inputs never leave the requester.
  {
    const route = new ShareRoute([new ProcessEvaluator(evaluatorCommand(), 'a'), new ProcessEvaluator(evaluatorCommand(), 'b')], seed, anchors);
    const tooWide = BigUint64Array.from(inputs, (value, index) => (index === 0 ? 1n << BigInt(BITS) : value));
    await assert.rejects(route.linear(NAME, tooWide, ROWS, OUTPUT_WIDTH, BITS), /outside the admitted share route/);
    await route.close();
  }

  // A provider that leaves mid-route ends the attempt with a bounded failure code.
  {
    const route = new ShareRoute([new ProcessEvaluator(evaluatorCommand(), 'a'), new ProcessEvaluator(['false'], 'b')], seed, anchors);
    await assert.rejects(route.linear(NAME, inputs, ROWS, OUTPUT_WIDTH, BITS), (error) => error instanceof ExecutionRejected && error.code === 'execution-failed');
  }

  // The same route through two enrolled providers on the authenticated, encrypted transport.
  {
    const providers = [];
    const publicRows = [];
    for (const [index, owner] of ['owner-a', 'owner-b'].entries()) {
      const computerId = `computer-${index}`;
      const privatePath = join(directory, `${computerId}.private.json`);
      const publicPath = join(directory, `${computerId}.public.json`);
      await new Promise((done, fail) => spawn(process.execPath, [SHARES_NETWORK_SCRIPT, 'keygen', '--computer-id', computerId,
        '--participant-id', `participant-${index}`, '--owner-id', owner, '--out-private', privatePath, '--out-public', publicPath],
      { stdio: 'inherit' }).once('exit', (code) => (code === 0 ? done() : fail(new Error(`keygen ${code}`)))));
      publicRows.push(JSON.parse(readFileSync(publicPath, 'utf8')));
      const server = spawn(process.execPath, [SHARES_NETWORK_SCRIPT, 'serve', '--listen', '127.0.0.1:0', '--identity', privatePath,
        '--cache-dir', directory, '--max-sessions', '1', '--', ...evaluatorCommand()], { stdio: ['ignore', 'pipe', 'inherit'] });
      const listening = await new Promise((done, fail) => {
        let text = '';
        server.stdout.on('data', (chunk) => {
          text += chunk;
          if (text.includes('\n')) done(JSON.parse(text.split('\n')[0]));
        });
        server.once('exit', (code) => fail(new Error(`serve exited ${code}`)));
      });
      providers.push({ server, computerId, endpoint: `${computerId}@127.0.0.1:${listening.port}`, manifestSha256: listening.manifestSha256 });
    }
    try {
      const enrollmentFile = join(directory, 'enrollment.json');
      writeFileSync(enrollmentFile, JSON.stringify({
        schema: 'doppler.two-server-shares-enrollment/v1',
        operatorIndependenceAttested: false,
        requesterOwnerId: 'requester',
        computers: publicRows,
      }));
      const manifestSha256 = createHash('sha256').update(manifestBytes).digest('hex');
      assert.equal(providers[0].manifestSha256, manifestSha256);
      const { commands, verification } = buildNetworkCommands({
        enrollmentFile,
        providerEndpoints: providers.map((provider) => provider.endpoint),
        planId: '1'.repeat(64),
        modelId: '2'.repeat(64),
        manifestSha256,
      });
      assert.equal(verification.distinctEnrolledOwnersVerified, true);
      assert.equal(verification.nonCollusionEstablished, false);
      const route = new ShareRoute(commands.map((command, index) => new ProcessEvaluator(command, 'ab'[index])), seed, anchors, { geometry });
      try {
        assert.deepEqual(await route.linear(NAME, inputs, ROWS, OUTPUT_WIDTH, BITS), expected);
      } finally {
        assert.deepEqual(await route.close(), [0, 0]);
      }
      route.requireIntegrity();
      assert.throws(() => buildNetworkCommands({
        enrollmentFile,
        providerEndpoints: [providers[0].endpoint, providers[0].endpoint],
        planId: '1'.repeat(64),
        modelId: '2'.repeat(64),
        manifestSha256,
      }), /Duplicate/);
    } finally {
      for (const provider of providers) provider.server.kill('SIGTERM');
    }
  }
} finally {
  rmSync(directory, { recursive: true, force: true });
}

console.log('shares-client.test: ok');
