import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const demoSource = await readFile(new URL('../../demo/demo-core.js', import.meta.url), 'utf8');
const compareConfig = JSON.parse(
  await readFile(new URL('../../benchmarks/vendors/compare-engines.config.json', import.meta.url), 'utf8')
);

assert.match(demoSource, /const DEFAULT_TRANSLATE_TARGET = 'es';/);
assert.match(demoSource, /const TRANSLATE_COMPARE_DEFAULT_TJS_DTYPE = 'q4';/);
assert.match(demoSource, /Object\.freeze\(\{ code: 'es', name: 'Spanish' \}\),/);
assert.doesNotMatch(demoSource, /Object\.freeze\(\{ code: 'es_XX', name: 'Spanish' \}\),/);
assert.match(
  demoSource,
  /id: 'proof'[\s\S]*left: Object\.freeze\(\{ engine: 'transformersjs', role: 'mapped-baseline' \}\),[\s\S]*right: Object\.freeze\(\{ engine: 'doppler', role: 'student' \}\),/
);
assert.match(
  demoSource,
  /TRANSLATE_COMPARE_TJS_BASELINE_NOTE = 'Baseline parity is currently unsupported in public TJS ONNX exports\.'/
);

const translategemmaProfile = (Array.isArray(compareConfig.modelProfiles) ? compareConfig.modelProfiles : [])
  .find((entry) => entry?.dopplerModelId === 'translategemma-4b-it-q4k-ehf16-af32');

assert.ok(translategemmaProfile, 'compare profiles must include the TranslateGemma baseline mapping');
assert.equal(translategemmaProfile.defaultTjsModelId, 'onnx-community/translategemma-text-4b-it-ONNX');
assert.equal(translategemmaProfile.defaultKernelPath, null);
assert.equal(translategemmaProfile.defaultDopplerSurface, 'auto');

console.log('translate-proof-demo-contract.test: ok');
