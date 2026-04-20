import assert from 'node:assert/strict';

// The top-level browser barrel intentionally exposes only `doppler` + `DOPPLER_VERSION`
// (see tools/check-public-boundaries.js). Structured/energy pipelines live under
// the dedicated `./structured` and `./experimental/energy` subpath exports.
const browserModule = await import('../../src/index-browser.js');
for (const name of ['DOPPLER_VERSION', 'doppler']) {
  assert.ok(browserModule[name] != null, `${name} should be exported by src/index-browser.js`);
}

const structured = await import('../../src/tooling-exports/structured.js');
for (const name of [
  'StructuredJsonHeadPipeline',
  'isStructuredJsonHeadModelType',
  'createStructuredJsonHeadPipeline',
  'DreamStructuredPipeline',
  'isDreamStructuredModelType',
  'createDreamStructuredPipeline',
]) {
  assert.equal(typeof structured[name], 'function', `${name} should be exported by doppler-gpu/structured`);
}

const energy = await import('../../src/inference/pipelines/energy-head/index.js');
for (const name of [
  'EnergyRowHeadPipeline',
  'createEnergyRowHeadPipeline',
  'DreamEnergyHeadPipeline',
  'createDreamEnergyHeadPipeline',
]) {
  assert.equal(typeof energy[name], 'function', `${name} should be exported by energy-head barrel`);
}

console.log('index-browser-export-surface.test: ok');
