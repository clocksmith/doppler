import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import { runImageTranscription } from '../../src/inference/browser-harness-text-helpers.js';

const samplePath = path.resolve(process.cwd(), 'examples/samples/sample-pastel-600x400.png');
const sampleBytes = await readFile(samplePath);

const server = createServer((req, res) => {
  if (req.url !== '/sample.png') {
    res.statusCode = 404;
    res.end('not found');
    return;
  }
  res.statusCode = 200;
  res.setHeader('Content-Type', 'image/png');
  res.end(sampleBytes);
});

await new Promise((resolve) => {
  server.listen(0, '127.0.0.1', resolve);
});

const address = server.address();
assert.ok(address && typeof address === 'object');
const imageUrl = `http://127.0.0.1:${address.port}/sample.png`;

const observed = {
  imageBytesLength: 0,
  width: 0,
  height: 0,
  prompt: null,
  maxTokens: 0,
  softTokenBudget: 0,
};

const pipeline = {
  tokenizer: {
    decode(ids) {
      const id = Array.isArray(ids) ? ids[0] : null;
      return id === 101 ? 'pastel' : '';
    },
  },
  async transcribeImage(input) {
    observed.imageBytesLength = input.imageBytes.length;
    observed.width = input.width;
    observed.height = input.height;
    observed.prompt = input.prompt;
    observed.maxTokens = input.maxTokens;
    observed.softTokenBudget = input.softTokenBudget;
    return {
      text: 'pastel',
      tokens: [101],
    };
  },
  getStats() {
    return {
      prefillTimeMs: 1,
      ttftMs: 1,
      decodeTimeMs: 1,
      prefillTokens: 1,
      decodeTokens: 1,
      decodeProfileSteps: [],
    };
  },
};

try {
  const result = await runImageTranscription(
    pipeline,
    {
      inference: {
        generation: {
          maxTokens: 12,
        },
      },
    },
    {
      prompt: 'Describe the image.',
      image: {
        url: imageUrl,
      },
      maxTokens: 12,
      softTokenBudget: 70,
    }
  );

  assert.equal(result.output, 'pastel');
  assert.equal(observed.width, 600);
  assert.equal(observed.height, 400);
  assert.equal(observed.imageBytesLength, 600 * 400 * 4);
  assert.equal(observed.prompt, 'Describe the image.');
  assert.equal(observed.maxTokens, 12);
  assert.equal(observed.softTokenBudget, 70);
} finally {
  await new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });
}

console.log('image-transcription-node-url.test: ok');
