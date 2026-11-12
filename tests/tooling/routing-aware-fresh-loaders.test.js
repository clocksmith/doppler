import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { instrumentMoeSharedSplitSource } from '../helpers/moe-shared-split-observation-loader.js';
import { instrumentPrefillLogitRowsSource } from '../helpers/prefill-logit-rows-loader.js';

// The routing-aware fresh-test observers patch exact source boundaries. They must stay
// unambiguous on the current runtime and only add an awaited observation call.
const read = relative => fs.readFile(new URL(`../../src/inference/pipelines/text/${relative}`, import.meta.url), 'utf8');

const moe = await read('ffn/moe.js');
const moeOut = instrumentMoeSharedSplitSource(moe);
assert.ok(moeOut.includes("Symbol.for('doppler.test.moe-shared-split-observation')"));
assert.equal(moeOut.replace(/\n  await globalThis\[Symbol\.for\('doppler\.test\.moe-shared-split-observation'\)\][\s\S]*?\}\);/, ''), moe);

const prefill = await read('generator/prefill-runtime.js');
const prefillOut = instrumentPrefillLogitRowsSource(prefill);
assert.ok(prefillOut.includes("Symbol.for('doppler.test.prefill-logit-rows')"));
assert.equal(prefillOut.replace(/        await globalThis\[Symbol\.for\('doppler\.test\.prefill-logit-rows'\)\][\s\S]*?\n        \}\);\n/, ''), prefill);

assert.throws(() => instrumentMoeSharedSplitSource('no boundary'), /unambiguous/);
assert.throws(() => instrumentPrefillLogitRowsSource('no boundary'), /unambiguous/);
console.log('routing-aware fresh loaders: ok');
