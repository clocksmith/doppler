import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';

const policyRaw = await fs.readFile('src/config/forge/reference/glimmer-30b-text.json');
const policy = JSON.parse(policyRaw.toString('utf8'));
const source = await fs.readFile('tools/glimmer-hf-text-reference.py', 'utf8');
const transcript = JSON.parse(await fs.readFile(policy.output, 'utf8'));
const acquisition = JSON.parse(await fs.readFile(policy.sourceAcquisitionReceipt, 'utf8'));
const sha256 = (value) => `sha256:${createHash('sha256').update(value).digest('hex')}`;
const acquisitionDigest = (filePath) => (
  `sha256:${acquisition.files.find((file) => file.path === filePath)?.observedSha256}`
);

assert.equal(policy.schema, 'doppler.pinned-transformers-text-reference/v1');
assert.equal(policy.revision, 'a4e59da52a7bc87ae7251dd5545c0dd437c44b68');
assert.equal(policy.transformersCommit, 'c7e57f79348480f73d3ef0ad8c47f807ef1378c8');
assert.equal(policy.generation.maxNewTokens, 128);
assert.equal(policy.generation.sampling, 'greedy-argmax-f32-logits');
assert.equal(policy.generation.useChatTemplate, false);
assert.equal(policy.execution.attentionImplementation, 'eager');
assert.equal(policy.author.kind, 'ai');
assert.match(source, /local_files_only=True/);
assert.match(source, /trust_remote_code=False/);
assert.match(source, /loadedTextParameters/);
assert.match(source, /preservedAuxiliaryParameters/);
assert.doesNotMatch(source, /\.to\(["']cuda["']\)/);

assert.equal(transcript.schema, 'doppler.pinned-source-transcript/v1');
assert.equal(transcript.model, policy.model);
assert.equal(transcript.revision, policy.revision);
assert.equal(transcript.entryPoint, 'text.generate');
assert.equal(transcript.prompt, policy.prompt);
assert.equal(transcript.generatedTokens, policy.generation.maxNewTokens);
assert.equal(transcript.generatedTokenIds.length, transcript.generatedTokens);
assert.equal(transcript.generatedTokenIds.every(Number.isInteger), true);
assert.equal(transcript.promptTokenIds.every(Number.isInteger), true);
assert.deepEqual(transcript.generation, policy.generation);
assert.equal(transcript.execution.device, 'cpu');
assert.equal(transcript.execution.dtype, 'bfloat16');
assert.equal(transcript.execution.attentionImplementation, 'eager');
assert.equal(transcript.loadEvidence.loadedTextParameters, 627);
assert.equal(transcript.loadEvidence.preservedAuxiliaryParameters, 809);
assert.equal(transcript.loadEvidence.weightShardCount, 2);
assert.equal(transcript.identity.policySha256, sha256(policyRaw));
assert.equal(transcript.identity.scriptSha256, sha256(source));
assert.equal(transcript.identity.promptSha256, sha256(policy.prompt));
assert.equal(transcript.identity.configSha256, acquisitionDigest('config.json'));
assert.equal(transcript.identity.generationConfigSha256, acquisitionDigest('generation_config.json'));
assert.equal(transcript.identity.tokenizerSha256, acquisitionDigest('tokenizer.json'));
assert.equal(transcript.identity.sourceAcquisitionReceiptDigest, acquisition.receiptDigest);
assert.equal(transcript.identity.transformersCommit, policy.transformersCommit);
assert.equal(transcript.boundaries.length, 9);
assert.equal(new Set(transcript.boundaries.map((entry) => entry.boundaryId)).size, 9);
for (const boundary of transcript.boundaries) {
  assert.equal(boundary.phase, 'prefill');
  assert.equal(boundary.dtype, 'float32');
  assert.equal(boundary.finite, true);
  assert.match(boundary.fullTensorDigest, /^sha256:[0-9a-f]{64}$/);
  assert.equal(boundary.elementCount > 0, true);
  assert.equal(Object.values(boundary.statistics).every(Number.isFinite), true);
}
assert.equal(transcript.decoded.withSpecialTokens.length > 0, true);
assert.equal(transcript.decoded.withoutSpecialTokens.length > 0, true);
assert.deepEqual(transcript.author, policy.author);

console.log('glimmer-hf-reference-contract.test: ok');
