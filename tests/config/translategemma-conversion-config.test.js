import assert from 'node:assert/strict';
import fs from 'node:fs';

const config = JSON.parse(
  fs.readFileSync(
    'tools/configs/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json',
    'utf8'
  )
);

assert.equal(config.inference?.schema, 'doppler.execution/v0');
assert.equal(config.inference?.sessionDefaults?.decodeLoop, null);

console.log('translategemma-conversion-config.test: ok');
