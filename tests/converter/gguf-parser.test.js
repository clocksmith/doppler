import assert from 'node:assert/strict';

import { parseGGUFModel } from '../../src/converter/parsers/gguf.js';

{
  const progress = [];
  let normalizedInput = null;
  let headerInput = null;
  const source = {
    file: { name: 'fixture.gguf', size: 1024 },
    size: 1024,
    readRange() {
      return new ArrayBuffer(0);
    },
  };

  const result = await parseGGUFModel({
    file: { name: 'fixture.gguf' },
    normalizeTensorSource(file) {
      normalizedInput = file;
      return source;
    },
    async parseGGUFHeaderFromSource(input) {
      headerInput = input;
      return {
        tensors: [
          {
            name: 'token_embd.weight',
            offset: 64,
            size: 128,
            dtype: 'F16',
            shape: [64, 2],
          },
        ],
        config: { model_type: 'transformer' },
        architecture: { numLayers: 1 },
        quantization: 'F16',
        tensorDataOffset: 64,
      };
    },
    onProgress(update) {
      progress.push(update);
    },
    signal: null,
  });

  assert.equal(normalizedInput?.name, 'fixture.gguf');
  assert.equal(headerInput, source);
  assert.equal(progress.length, 1);
  assert.equal(result.format, 'gguf');
  assert.equal(result.file, source.file);
  assert.equal(result.source, source);
  assert.equal(result.tensorDataOffset, 64);
  assert.equal(result.quantization, 'F16');
  assert.equal(result.tensors.length, 1);
  assert.equal(result.tensors[0].name, 'token_embd.weight');
  assert.equal(result.tensors[0].offset, 64);
  assert.equal(result.tensors[0].source, source);
  assert.equal(result.tensors[0].file, source.file);
}

{
  let parseCalled = false;
  await assert.rejects(
    () => parseGGUFModel({
      file: { name: 'cancel.gguf' },
      normalizeTensorSource() {
        return {
          file: { name: 'cancel.gguf', size: 8 },
          size: 8,
          readRange() {
            return new ArrayBuffer(0);
          },
        };
      },
      async parseGGUFHeaderFromSource() {
        parseCalled = true;
        return null;
      },
      signal: { aborted: true },
    }),
    (error) => {
      assert.equal(error?.name, 'AbortError');
      return true;
    }
  );
  assert.equal(parseCalled, false);
}

console.log('gguf-parser.test: ok');
