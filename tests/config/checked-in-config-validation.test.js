import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';

import { listRuntimeProfiles } from '../../src/cli/runtime-profiles.js';
import { resolveRuntimeConfig } from '../../src/config/runtime.js';
import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import { loadRuntimeProfile } from '../../src/inference/browser-harness/runtime-config.js';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { stripLegacyRuntimeIntent } from '../../src/tooling/runtime-intent.js';

// Every checked-in runtime profile and conversion config goes through the real
// load/compile path. Per-file value pins are intentionally absent.

installNodeFileFetchShim();

const { profiles } = await listRuntimeProfiles();
assert.ok(profiles.length > 0, 'runtime profile discovery must find checked-in profiles');
for (const profile of profiles) {
  const profileRef = profile.runtimePath.replace(/\.json$/u, '');
  const loaded = await loadRuntimeProfile(profileRef);
  assert.doesNotThrow(
    () => resolveRuntimeConfig(stripLegacyRuntimeIntent(loaded.runtime)),
    `${profile.path} must resolve through extends and runtime validation`
  );
}

function collectJsonFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.join(dir, entry.name);
    if (entry.isDirectory()) return collectJsonFiles(entryPath);
    return entry.isFile() && entry.name.endsWith('.json') ? [entryPath] : [];
  });
}

const conversionFiles = collectJsonFiles('src/config/conversion');
assert.ok(conversionFiles.length > 0, 'conversion config discovery must find checked-in configs');
for (const filePath of conversionFiles) {
  const config = JSON.parse(readFileSync(filePath, 'utf8'));
  let steps;
  assert.doesNotThrow(
    () => { steps = expandExecutionV1(config.execution); },
    `${filePath} execution graph must compile`
  );
  assert.ok(steps.length > 0, `${filePath} execution graph must expand to at least one step`);
}

console.log('checked-in-config-validation.test: ok');
