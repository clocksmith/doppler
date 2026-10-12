// Diagnostic intervention only. These archives are inadmissible for qualification.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { resolve, join } from 'node:path';
const [archive, fixturePath, modelPath, output, mode] = process.argv.slice(2);
assert(['captured-control', 'independent-local'].includes(mode));
const hash = value => createHash('sha256').update(value).digest('hex');
const parentArchiveSha256 = hash(await readFile(archive));
assert.equal(parentArchiveSha256, 'c596ecab6d498b3bb6e4d070e65a640713e693020e77c897271481f197875aae');
const fixtureBytes = gunzipSync(await readFile(fixturePath));
const fixture = JSON.parse(fixtureBytes);
assert.equal(fixture.captureSha256, '4d377a97fd0f1d4aa3f9a4e92900a35fa3b8b33bca0846e72c1e9e85dc731d18');
assert.equal(fixture.layer, 15);
assert.equal(fixture.seqLen, 184);
const values = Float32Array.from(mode === 'captured-control' ? fixture.observedF32 : fixture.float64Expected);
const encoded = Buffer.from(values.buffer).toString('base64');
await mkdir(output);
await promisify(execFile)('tar', ['-xzf', resolve(archive), '-C', resolve(output)]);
const root = resolve(output, 'package');
const path = join(root, 'src/inference/pipelines/text/probes.js');
let source = await readFile(path, 'utf8');
const anchor = "  const { layerIdx, numTokens, hiddenSize, probes, recorder, dtype = 'f32' } = options;";
assert.equal(source.split(anchor).length, 2);
source = source.replace(anchor, anchor + `
  // Explicit retained diagnostic intervention; never shipped as a qualified runtime.
  if (stage === 'attn_core_out' && layerIdx === 15 && numTokens === 184) {
    if (dtype !== 'f32' || hiddenSize !== ${fixture.numHeads * fixture.headDim}) throw new Error('Counterfactual geometry mismatch');
    const bytes = Uint8Array.from(atob('${encoded}'), character => character.charCodeAt(0));
    const device = getDevice();
    if (recorder) {
      const staging = device.createBuffer({ label: 'independent_attention_counterfactual', size: bytes.byteLength,
        usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
      new Uint8Array(staging.getMappedRange()).set(bytes); staging.unmap();
      recorder.getEncoder().copyBufferToBuffer(staging, 0, buffer, 0, bytes.byteLength);
      recorder.trackTemporaryBuffer(staging);
    } else { device.queue.writeBuffer(buffer, 0, bytes); }
    console.log('producer-progress:' + JSON.stringify({ phase: 'counterfactual', mode: '${mode}',
      layer: 15, numTokens, bytes: bytes.byteLength, outputSha256: '${hash(Buffer.from(values.buffer))}', qualification: false }));
  }`);
await writeFile(path, source);
const pack = await promisify(execFile)('npm', ['pack', '--ignore-scripts', '--pack-destination', resolve(output)], { cwd: root });
await writeFile(join(output, 'pack.log'), pack.stdout + pack.stderr);
const file = pack.stdout.trim().split('\n').at(-1);
const binding = { archiveSha256: hash(await readFile(join(output, file))), parentArchiveSha256,
  manifestSha256: hash(await readFile(join(modelPath, 'manifest.json'))),
  sourceManifestSha256: '18fe30e2c65c47bfdaa1f6714ced9fa90ed8c482a8d57b95cf0cdd4490e842e9',
  fixtureSha256: hash(fixtureBytes), captureSha256: fixture.captureSha256,
  injectedOutputSha256: hash(Buffer.from(values.buffer)), mode, unchangedWeights: true, tolerance: 0.001,
  scope: 'Only prefill layer15 attention core overwritten, followed by real GPU downstream; explicitly inadmissible for qualification' };
await writeFile(join(output, 'binding.json'), JSON.stringify(binding, null, 2) + '\n');
console.log(JSON.stringify(binding));
