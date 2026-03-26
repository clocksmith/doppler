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

console.log('rdrr-parsing-contract.test: ok');
