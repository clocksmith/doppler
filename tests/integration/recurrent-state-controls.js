import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { runRecurrentStateControl } from '../fixtures/recurrent-state-control-browser.js';
import { pairLinearCaptures, recurrentReference, decodeCapturedTensor, RECURRENT_REFERENCE } from '../kernels/recurrent-reference.js';
import { recurrentReferenceStep } from '../kernels/recurrent-reference-step.js';
import { buildRecurrentAccumulationDiagnostic } from '../kernels/recurrent-accumulation-diagnostic.js';

const [capturePath, installed, reploid, destination] = process.argv.slice(2);
assert(destination && process.env.REPLOID_EXECUTOR_WS);
assert.equal(process.env.DOPPLER_TEST_ONLY_ARITHMETIC, '1', 'Reference-state controls are diagnostic only');
const bytes = await readFile(capturePath), capture = JSON.parse(bytes);
const pairs = pairLinearCaptures(capture); assert.equal(pairs.length, 1);
const { input, expected, coordinate } = pairs[0], p = input.params;
const reference = recurrentReference(input, expected), field = reference.layout.fields.updatedState;
const referenceStates = Float32Array.from(reference.trace.subarray(field.offset, field.offset + field.length));
const originalShader = await readFile(resolve(installed, 'src/gpu/kernels/gated_delta_recurrent.wgsl'), 'utf8');
const candidate = process.env.DOPPLER_RECURRENT_DOT_CANDIDATE || null;
assert(!candidate || ['memory', 'readout'].includes(candidate), 'Evaluate one dot correction at a time');
const shader = candidate ? buildRecurrentAccumulationDiagnostic(originalShader, candidate) : originalShader;
const hash = b => createHash('sha256').update(b).digest('hex');
const { chromium } = createRequire(resolve(reploid, 'package.json'))('playwright');
const receipt = { scope: 'Reference-reset versus continuous GPU recurrent state; captured upstream operands are frozen',
  archiveSha256: capture.archiveSha256, captureSha256: hash(bytes), coordinate,
  shaderSha256: hash(shader), originalShaderSha256: hash(originalShader), sourceSubstitution: Boolean(candidate), candidate,
  reference: RECURRENT_REFERENCE, completed: false, results: [] };
await writeFile(`${destination}.wgsl`, shader);
const values = text => { const bytes = Buffer.from(text, 'base64'); return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4); };
const compare = (a, b) => {
  assert.equal(a.length, b.length); let square = 0, maxDifference = 0;
  for (let i = 0; i < a.length; i++) {
    assert(Number.isFinite(a[i]) && Number.isFinite(b[i]));
    const difference = Math.abs(a[i] - b[i]); square += difference ** 2; maxDifference = Math.max(maxDifference, difference);
  }
  return { elements: a.length, maxDifference, rmsError: Math.sqrt(square / a.length) };
};
try {
  for (const platform of ['mac', 'linux']) {
    const browser = platform === 'mac'
      ? await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] })
      : await chromium.connect(process.env.REPLOID_EXECUTOR_WS);
    try {
      for (const mode of ['reference-reset', 'continuous']) {
        const context = await browser.newContext();
        try {
          const page = await context.newPage(); await page.goto('http://localhost:8000/config/chat-files.json');
          if (mode === 'reference-reset') {
            const text = Buffer.from(referenceStates.buffer).toString('base64');
            await page.evaluate(() => { globalThis.referenceStateChunks = []; });
            for (let i = 0; i < text.length; i += 262144) await page.evaluate(chunk => {
              globalThis.referenceStateChunks.push(chunk);
            }, text.slice(i, i + 262144));
          }
          let timer;
          const result = await Promise.race([page.evaluate(runRecurrentStateControl, { input, expected, shader, mode }),
            new Promise((_, reject) => { timer = setTimeout(() => reject(Error('State control exceeded 60 seconds')), 60000); })])
            .finally(() => clearTimeout(timer));
          const stateSize = reference.finalState.length;
          let prior = decodeCapturedTensor(input, 'recurrentState').subarray(0, stateSize);
          for (const row of result.tokens) {
            const token = row.token;
            const reset = token === 0 ? decodeCapturedTensor(input, 'recurrentState').subarray(0, stateSize)
              : referenceStates.subarray((token - 1) * stateSize, token * stateSize);
            const local = recurrentReferenceStep(input, expected, token, mode === 'reference-reset' ? reset : prior);
            row.localArithmetic = { output: compare(values(row.output), local.output), state: compare(values(row.state), local.finalState) };
            row.totalTrajectory = { output: compare(values(row.output), reference.output.subarray(token * p.valueDim, (token + 1) * p.valueDim)),
              state: compare(values(row.state), reference.trace.subarray(field.offset + token * stateSize, field.offset + (token + 1) * stateSize)) };
            prior = values(row.state);
          }
          assert.equal(result.stateWrites, mode === 'reference-reset' ? p.numTokens : 1);
          receipt.results.push({ platform, mode, browser: browser.version(), ...result });
          await writeFile(destination, JSON.stringify(receipt));
          console.log(JSON.stringify({ platform, mode, tokens: result.tokens.length,
            last: { local: result.tokens.at(-1).localArithmetic, total: result.tokens.at(-1).totalTrajectory } }));
        } finally { await context.close(); }
      }
    } finally { await browser.close(); }
  }
  receipt.crossPlatform = ['reference-reset', 'continuous'].map(mode => {
    const pair = receipt.results.filter(r => r.mode === mode);
    return { mode, tokens: pair[0].tokens.map((row, i) => ({ token: i,
      output: compare(values(row.output), values(pair[1].tokens[i].output)), state: compare(values(row.state), values(pair[1].tokens[i].state)) })) };
  });
  receipt.completed = true;
} catch (error) {
  receipt.failure = error.stack;
  throw error;
} finally {
  await writeFile(destination, JSON.stringify(receipt));
}
