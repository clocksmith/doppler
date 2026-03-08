import assert from 'node:assert/strict';

const browserModule = await import('../../src/index-browser.js');

for (const name of [
  'StructuredJsonHeadPipeline',
  'isStructuredJsonHeadModelType',
  'createStructuredJsonHeadPipeline',
  'DreamStructuredPipeline',
  'isDreamStructuredModelType',
  'createDreamStructuredPipeline',
  'EnergyRowHeadPipeline',
  'createEnergyRowHeadPipeline',
  'DreamEnergyHeadPipeline',
  'createDreamEnergyHeadPipeline',
]) {
  assert.equal(typeof browserModule[name], 'function', `${name} should be exported by src/index-browser.js`);
}

console.log('index-browser-export-surface.test: ok');
