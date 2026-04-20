import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { parseManifest } from '../../src/formats/rdrr/parsing.js';

function clone(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

const canonicalManifest = JSON.parse(
  readFileSync(
    new URL('../../models/local/gemma-3-1b-it-q4k-ehf16-af32/manifest.json', import.meta.url),
    'utf8'
  )
);

{
  const missingSession = clone(canonicalManifest);
  delete missingSession.inference.session;
  assert.throws(
    () => parseManifest(JSON.stringify(missingSession)),
    /Invalid manifest:/,
    'transformer manifests must not parse without inference.session'
  );
}

{
  const legacySessionDefaults = clone(canonicalManifest);
  legacySessionDefaults.inference.sessionDefaults = legacySessionDefaults.inference.session;
  delete legacySessionDefaults.inference.session;
  assert.throws(
    () => parseManifest(JSON.stringify(legacySessionDefaults)),
    /Invalid manifest:/,
    'parser must not silently promote sessionDefaults to session'
  );
}

{
  const missingDecodeLoop = clone(canonicalManifest);
  delete missingDecodeLoop.inference.session.decodeLoop;
  assert.throws(
    () => parseManifest(JSON.stringify(missingDecodeLoop)),
    /Invalid manifest:/,
    'transformer manifests must not parse without inference.session.decodeLoop'
  );
}

{
  const identityManifest = clone(canonicalManifest);
  identityManifest.artifactIdentity = {
    sourceCheckpointId: 'google/gemma-3-1b-it@example',
    sourceRepo: 'google/gemma-3-1b-it',
    sourceRevision: 'example',
    sourceFormat: 'safetensors',
    conversionConfigPath: 'src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json',
    conversionConfigDigest: 'sha256:conversion-config',
    weightPackId: 'gemma3-1b-text-q4k-ehf16-v1',
    weightPackHash: 'sha256:weight-pack',
    manifestVariantId: 'gemma3-1b-text-q4k-ehf16-af32-exec-v1',
    modalitySet: ['text'],
    materializationProfile: 'standard',
    artifactCompleteness: 'complete',
  };
  identityManifest.weightsRef = {
    weightPackId: 'gemma3-1b-text-q4k-ehf16-v1',
    artifactRoot: 'models/local/gemma-3-1b-it-q4k-ehf16-af32',
    manifestDigest: 'sha256:manifest',
    shardSetHash: 'sha256:shard-set',
  };
  const parsed = parseManifest(JSON.stringify(identityManifest));
  assert.deepEqual(parsed.artifactIdentity, identityManifest.artifactIdentity);
  assert.deepEqual(parsed.weightsRef, identityManifest.weightsRef);
}

{
  const legacyIdentityManifest = clone(canonicalManifest);
  delete legacyIdentityManifest.artifactIdentity;
  delete legacyIdentityManifest.weightsRef;
  const parsed = parseManifest(JSON.stringify(legacyIdentityManifest));
  assert.equal(parsed.artifactIdentity, undefined);
  assert.equal(parsed.weightsRef, undefined);
}

{
  const invalidIdentityManifest = clone(canonicalManifest);
  invalidIdentityManifest.artifactIdentity = {
    weightPackId: '',
  };
  assert.throws(
    () => parseManifest(JSON.stringify(invalidIdentityManifest)),
    /Invalid artifactIdentity\.weightPackId/,
    'artifact identity fields must be non-empty when present'
  );
}

{
  const invalidWeightsRefManifest = clone(canonicalManifest);
  invalidWeightsRefManifest.weightsRef = {
    weightPackId: 'gemma3-1b-text-q4k-ehf16-v1',
    artifactRoot: 'models/local/gemma-3-1b-it-q4k-ehf16-af32',
    shardSetHash: 'sha256:shard-set',
  };
  assert.throws(
    () => parseManifest(JSON.stringify(invalidWeightsRefManifest)),
    /Missing or invalid weightsRef\.manifestDigest/,
    'weightsRef must carry enough identity to resolve a shared weight pack explicitly'
  );
}

console.log('rdrr-parsing-contract.test: ok');
