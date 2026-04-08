import fs from 'node:fs';

import {
  assertManifestArtifactIntegrity,
  readLocalFixtureMap,
} from '../helpers/local-model-fixture.js';

const fixtures = readLocalFixtureMap();
let checked = 0;

for (const entry of fixtures) {
  if (!fs.existsSync(entry.manifestPath)) {
    continue;
  }
  await assertManifestArtifactIntegrity(entry.manifestPath);
  checked += 1;
}

if (checked === 0) {
  console.log('local-model-fixture-integrity.test: skipped (no local fixture manifests present)');
} else {
  console.log(`local-model-fixture-integrity.test: ok (${checked} local manifests checked)`);
}
