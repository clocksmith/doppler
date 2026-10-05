// Prepare a separate test manifest from pinned source bytes and a conversion contract.
// Weight bytes, model mathematics and source files are never modified.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, link, realpath } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve, join, sep } from 'node:path';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

const [sourceArg, contractPath, conversionPath, outputArg] = process.argv.slice(2);
assert(outputArg, 'Expected source model, frozen source contract, conversion contract and new output directory');
const source = resolve(sourceArg), output = resolve(outputArg);
assert.notEqual(source, output);
const bytes = await readFile(join(source, 'manifest.json'));
const hash = value => createHash('sha256').update(value).digest('hex');
const contract = JSON.parse(await readFile(contractPath, 'utf8'));
assert.equal(hash(bytes), contract.manifestSha256, 'Source manifest differs from frozen input');
const manifest = JSON.parse(bytes), conversion = JSON.parse(await readFile(conversionPath, 'utf8'));
assert.equal(manifest.modelId, contract.modelId);
assert(Array.isArray(conversion.execution.mechanismKernels), 'Explicit loading/mechanism contract required');
const changes = [];
for (const [id, declaration] of Object.entries(manifest.inference.execution.kernels)) {
  const current = KERNEL_REF_CONTENT_DIGESTS[`${declaration.kernel}#${declaration.entry}`];
  assert(current, `Unknown declared shader ${id}`);
  if (declaration.digest !== 'sha256:' + current) {
    changes.push({ id, previous: declaration.digest, current: 'sha256:' + current });
    declaration.digest = 'sha256:' + current;
  }
}
for (const id of conversion.execution.mechanismKernels) {
  const declaration = structuredClone(conversion.execution.kernels[id]);
  assert(declaration, `Missing declared mechanism ${id}`);
  const current = KERNEL_REF_CONTENT_DIGESTS[`${declaration.kernel}#${declaration.entry}`];
  assert(current, `Unknown mechanism shader ${id}`);
  declaration.digest = 'sha256:' + current;
  manifest.inference.execution.kernels[id] = declaration;
}
manifest.inference.execution.mechanismKernels = conversion.execution.mechanismKernels;
manifest.manifestVariantId = manifest.modelId + '-choice-source-closure-v1';
const derivedBytes = JSON.stringify(manifest, null, 2) + '\n';
await mkdir(output); // Refuse to overwrite an existing experiment.
await writeFile(join(output, 'manifest.json'), derivedBytes);
const artifacts = [];
for (const name of [...manifest.shards.map(shard => shard.filename), manifest.tokenizer.file]) {
  const original = resolve(source, name);
  assert(original.startsWith(source + sep), 'Artifact path escapes the source model');
  const originalResolved = await realpath(original);
  await link(originalResolved, join(output, name));
  const digest = createHash('sha256'); let sizeBytes = 0;
  for await (const chunk of createReadStream(originalResolved)) { digest.update(chunk); sizeBytes += chunk.length; }
  artifacts.push({ name, sha256: digest.digest('hex'), sizeBytes });
}
const derivedContract = { ...contract, manifestSha256: hash(derivedBytes) };
await writeFile(join(output, 'choice-contract.json'), JSON.stringify(derivedContract, null, 2) + '\n');
await writeFile(join(output, 'derivation.json'), JSON.stringify({ schema: 'doppler.test-manifest-derivation/v1',
  originalManifestSha256: hash(bytes), candidateManifestSha256: hash(derivedBytes),
  conversionContractSha256: hash(await readFile(conversionPath)), changes,
  mechanismKernels: conversion.execution.mechanismKernels, artifacts,
  weightHandling: 'Unmodified hard links; source manifest unchanged.' }, null, 2) + '\n');
console.log(JSON.stringify({ output, manifestSha256: derivedContract.manifestSha256 }));
