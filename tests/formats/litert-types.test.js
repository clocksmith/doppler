import assert from 'node:assert/strict';
import {
  FIXTURE_TFLITE_TENSOR_TYPE,
  buildTfliteFixture,
} from '../helpers/tflite-fixture.js';
import {
  FIXTURE_LITERTLM_SECTION_TYPE,
  buildLiteRTTaskFixture,
  buildLiteRTLmFixture,
} from '../helpers/litert-package-fixture.js';

const {
  LITERT_TASK_DEFAULT_TFLITE_ENTRY,
  LITERT_TASK_DEFAULT_TOKENIZER_MODEL_ENTRY,
  LITERTLM_MAGIC,
  findLiteRTLMMetadataSection,
  findLiteRTLMSentencePieceTokenizerSection,
  findLiteRTLMTFLiteModelSection,
  findLiteRTLMTFLiteWeightsSection,
  parseLiteRTLMFromSource,
  parseLiteRTTaskFromSource,
} = await import('../../src/formats/litert/types.js');

function createMemorySource(name, bytes) {
  return {
    name,
    size: bytes.byteLength,
    async readRange(offset, length) {
      const start = Math.max(0, Math.floor(offset));
      const end = Math.min(bytes.byteLength, start + Math.max(0, Math.floor(length)));
      return bytes.slice(start, end);
    },
  };
}

const tfliteBytes = buildTfliteFixture({
  description: 'litert-format-fixture',
  tensors: [
    {
      name: 'model.embed_tokens.weight',
      shape: [2, 2],
      type: FIXTURE_TFLITE_TENSOR_TYPE.FLOAT16,
      data: Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 7]),
    },
  ],
});

const taskBytes = buildLiteRTTaskFixture([
  { name: LITERT_TASK_DEFAULT_TFLITE_ENTRY, data: tfliteBytes },
  { name: LITERT_TASK_DEFAULT_TOKENIZER_MODEL_ENTRY, data: Uint8Array.from([1, 2, 3, 4]) },
  { name: 'METADATA', data: Uint8Array.from([5, 6, 7]) },
]);

const parsedTask = await parseLiteRTTaskFromSource(createMemorySource('fixture.task', taskBytes));
assert.equal(parsedTask.entries.length, 3);
assert.ok(parsedTask.entryMap.has(LITERT_TASK_DEFAULT_TFLITE_ENTRY));
assert.equal(parsedTask.entryMap.get(LITERT_TASK_DEFAULT_TFLITE_ENTRY)?.size, tfliteBytes.byteLength);

const litertlmBytes = buildLiteRTLmFixture({
  sections: [
    {
      dataType: FIXTURE_LITERTLM_SECTION_TYPE.TFLiteModel,
      data: tfliteBytes,
    },
    {
      dataType: FIXTURE_LITERTLM_SECTION_TYPE.SP_Tokenizer,
      data: Uint8Array.from([9, 8, 7, 6]),
    },
    {
      dataType: FIXTURE_LITERTLM_SECTION_TYPE.LlmMetadataProto,
      data: Uint8Array.from([1, 1, 2, 3]),
    },
  ],
});

assert.equal(new TextDecoder().decode(litertlmBytes.slice(0, 8)), LITERTLM_MAGIC);
const parsedLiteRTLM = await parseLiteRTLMFromSource(createMemorySource('fixture.litertlm', litertlmBytes));
assert.equal(parsedLiteRTLM.majorVersion, 1);
assert.equal(parsedLiteRTLM.sections.length, 3);
assert.equal(findLiteRTLMTFLiteModelSection(parsedLiteRTLM)?.dataTypeName, 'TFLiteModel');
assert.equal(findLiteRTLMSentencePieceTokenizerSection(parsedLiteRTLM)?.dataTypeName, 'SP_Tokenizer');
assert.equal(findLiteRTLMMetadataSection(parsedLiteRTLM)?.dataTypeName, 'LlmMetadataProto');
assert.equal(findLiteRTLMTFLiteWeightsSection(parsedLiteRTLM), null);

console.log('litert-types.test: ok');
