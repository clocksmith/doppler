import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { installMoeRoutingObservation } from '../helpers/moe-routing-observation-loader.js';

const directory = await mkdtemp(path.join(tmpdir(), 'doppler-moe-observation-'));
const source = `export async function route() {
  const layerIdx = 1, numTokens = 3, topK = 2, activationDtype = 'f32';
  const indicesBuffer = { name: 'indices' }, weightsBuffer = { name: 'weights' };
  const stepStart = 0, perfLog = () => {};
  perfLog(\`MoE L\${layerIdx} topk\`, stepStart, {
    topK,
  });
  return { indicesBuffer, weightsBuffer };
}`;
const hash = value => createHash('sha256').update(value).digest('hex');
let count = 0;
async function fixture(text = source) {
  const file = path.join(directory, `${count++}.js`);
  await writeFile(file, text);
  return pathToFileURL(await realpath(file)).href;
}

try {
  let captured;
  const url = await fixture();
  const observer = installMoeRoutingObservation(url, hash(source), value => { captured = value; });
  try {
    const result = await (await import(url)).route();
    assert.equal(result.indicesBuffer, captured.indicesBuffer);
    assert.equal(result.weightsBuffer, captured.weightsBuffer);
    assert.equal(captured.layer, 1);
    assert.equal(observer.receipt.originalSha256, hash(source));
    assert.notEqual(observer.receipt.instrumentedSha256, hash(source));
  } finally { observer.restore(); }

  const wrong = await fixture();
  const rejected = installMoeRoutingObservation(wrong, '0'.repeat(64), () => {});
  try { await assert.rejects(import(wrong), /AssertionError/); }
  finally { rejected.restore(); }

  const absent = await fixture('export const noRouting = true;');
  const missing = installMoeRoutingObservation(absent, hash('export const noRouting = true;'), () => {});
  try { await assert.rejects(import(absent), /unambiguous original top-k/); }
  finally { missing.restore(); }

  const failing = await fixture();
  const failure = new Error('readback failed');
  const broken = installMoeRoutingObservation(failing, hash(source), () => { throw failure; });
  try { await assert.rejects((await import(failing)).route(), error => error === failure); }
  finally { broken.restore(); }
  assert.equal(globalThis[Symbol.for('doppler.test.moe-routing-observation')], undefined);
  console.log('MoE routing observations preserve buffers, reject wrong source, and release ownership on failure.');
} finally { await rm(directory, { recursive: true, force: true }); }
