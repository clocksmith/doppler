import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const modulePath = new URL('../../benchmarks/vendors/workload-prompt.js', import.meta.url);
const moduleSource = await fs.readFile(modulePath, 'utf8');

assert.match(moduleSource, /const localModelRoot = path\.resolve\(localModelPath\);/);
assert.match(moduleSource, /const localModelDir = normalizedModelId\s*\?\s*path\.join\(localModelRoot, normalizedModelId\)/);
assert.match(moduleSource, /if \(localModelDir && fs\.existsSync\(localModelDir\)\)/);
assert.match(moduleSource, /source: 'local-model-root'/);
assert.match(moduleSource, /tokenizerResolutionSource = `\$\{primaryLocator\.source\}-fallback-model-id`/);
assert.match(moduleSource, /const chatTemplatePath = path\.join\(tokenizerLocator, 'chat_template\.jinja'\);/);
assert.match(moduleSource, /chatTemplateOverride \? \{ chat_template: chatTemplateOverride } : \{\}/);

console.log('workload-prompt-local-model-path-contract.test: ok');
