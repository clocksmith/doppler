import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const demoSource = readFileSync(new URL('../../demo/demo-core.js', import.meta.url), 'utf8');
const downloadsSource = readFileSync(new URL('../../demo/ui/downloads/index.js', import.meta.url), 'utf8');
const storageInspectorSource = readFileSync(new URL('../../demo/ui/storage/inspector.js', import.meta.url), 'utf8');
const diagnosticsSource = readFileSync(new URL('../../demo/diagnostics-controller.js', import.meta.url), 'utf8');

assert.match(downloadsSource, /listStorageInventory/);
assert.match(downloadsSource, /deleteStorageEntry/);
assert.match(downloadsSource, /async function cleanupPartialImport\(modelId\)/);
assert.match(downloadsSource, /await cleanupPartialImport\(working\.modelId\)/);

assert.match(demoSource, /runtimeConfig: getRuntimeConfig\(\),/);
assert.match(demoSource, /function isRunnableStorageEntry\(entry\)/);
assert.match(demoSource, /const isInOpfs = isRunnableStorageEntry\(storageEntry\)/);
assert.match(demoSource, /Imported \$\{modelId\} to \$\{describeImportedStorage\(modelId\)\}\./);
assert.doesNotMatch(demoSource, /Imported \$\{modelId\} to OPFS\./);

assert.match(storageInspectorSource, /if \(entry\.hasManifest\) \{/);

assert.match(diagnosticsSource, /does not accept configChain/);
assert.match(diagnosticsSource, /runtimeProfile:\s*options\.runtimeProfile\s*\?\?\s*null/);

console.log('demo-surface-contract.test: ok');
