import assert from 'node:assert/strict';

import {
  formatRerankPrompt,
  resolveRerankScoringConfig,
  runRerank,
  runRerankSemanticChecks,
  scoreRerankDocument,
} from '../../src/inference/browser-harness-text-helpers.js';

const scoringConfig = {
  format: 'qwen3_yes_no_logit',
  instruction: 'Given a web search query, retrieve relevant passages that answer the query',
  inputTemplate: '<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}',
  prefix: '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
  suffix: '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
  trueToken: 'yes',
  trueTokenId: 2,
  falseToken: 'no',
  falseTokenId: 3,
  score: 'logit_difference',
  probability: 'sigmoid',
};

const positiveSnippets = [
  'Beijing',
  'forgot-password',
  'sorted()',
  'WebGPU',
  'immune system',
];

const pipeline = {
  manifest: {
    inference: {
      supportsRerank: true,
      rerank: scoringConfig,
    },
  },
  runtimeConfig: {},
  resetCount: 0,
  reset() {
    this.resetCount++;
  },
  async prefillWithLogits(prompt, options) {
    assert.equal(options.useChatTemplate, false);
    const logits = new Float32Array(4);
    const isPositive = positiveSnippets.some((snippet) => prompt.includes(snippet));
    logits[scoringConfig.trueTokenId] = isPositive ? 4 : -2;
    logits[scoringConfig.falseTokenId] = isPositive ? -2 : 4;
    return {
      tokens: Array.from({ length: Math.max(1, Math.min(8, Math.floor(prompt.length / 20))) }, (_, i) => i + 1),
      logits,
    };
  },
};

const prompt = formatRerankPrompt(
  'What is the capital of China?',
  'The capital of China is Beijing.',
  scoringConfig
);
assert.ok(prompt.startsWith(scoringConfig.prefix));
assert.ok(prompt.includes('<Query>: What is the capital of China?'));
assert.ok(prompt.includes('<Document>: The capital of China is Beijing.'));
assert.ok(prompt.endsWith(scoringConfig.suffix));

const score = await scoreRerankDocument(
  pipeline,
  'What is the capital of China?',
  'The capital of China is Beijing.'
);
assert.equal(score.trueTokenId, 2);
assert.equal(score.falseTokenId, 3);
assert.equal(score.score, 6);
assert.equal(score.scoringPath, 'full-logits');
assert.ok(score.probability > 0.99);
assert.equal(pipeline.resetCount, 1);

const selectedLogitPipeline = {
  manifest: {
    inference: {
      supportsRerank: true,
      rerank: scoringConfig,
    },
  },
  resetCount: 0,
  fullLogitCalls: 0,
  selectedLogitCalls: 0,
  reset() {
    this.resetCount++;
  },
  async prefillWithTokenLogits(prompt, tokenIds, options) {
    assert.equal(options.useChatTemplate, false);
    assert.deepEqual(tokenIds, [scoringConfig.trueTokenId, scoringConfig.falseTokenId]);
    this.selectedLogitCalls++;
    const isPositive = positiveSnippets.some((snippet) => prompt.includes(snippet));
    return {
      tokens: [1, 2, 3],
      tokenIds,
      logits: Float32Array.from(isPositive ? [7, -1] : [-1, 7]),
      logitsByTokenId: {
        [scoringConfig.trueTokenId]: isPositive ? 7 : -1,
        [scoringConfig.falseTokenId]: isPositive ? -1 : 7,
      },
    };
  },
  async prefillWithLogits() {
    this.fullLogitCalls++;
    throw new Error('full logits should not be used when selected-token logits are available');
  },
};

const selectedScore = await scoreRerankDocument(
  selectedLogitPipeline,
  'What is the capital of China?',
  'The capital of China is Beijing.'
);
assert.equal(selectedScore.score, 8);
assert.equal(selectedScore.scoringPath, 'selected-token-logits');
assert.equal(selectedLogitPipeline.resetCount, 1);
assert.equal(selectedLogitPipeline.selectedLogitCalls, 1);
assert.equal(selectedLogitPipeline.fullLogitCalls, 0);

const prefixPipeline = {
  manifest: {
    inference: {
      supportsRerank: true,
      rerank: scoringConfig,
    },
  },
  tokenizer: {
    encode(text) {
      return Array.from(text, (char) => char.charCodeAt(0));
    },
  },
  resetCount: 0,
  prefillKVOnlyCalls: 0,
  prefixSelectedCalls: 0,
  currentSeqLen: 0,
  reset() {
    this.resetCount++;
    this.currentSeqLen = 0;
  },
  resetToSeqLen(seqLen) {
    this.currentSeqLen = seqLen;
  },
  async prefillKVOnly(prompt, options) {
    assert.equal(prompt, '');
    assert.equal(options.useChatTemplate, false);
    assert.ok(Array.isArray(options.inputIds));
    assert.ok(options.inputIds.length > 0);
    this.prefillKVOnlyCalls++;
    this.currentSeqLen = options.inputIds.length;
    return {
      cache: {
        clone() {
          return { setGPUContext() {} };
        },
      },
      seqLen: options.inputIds.length,
      tokens: options.inputIds,
      linearAttention: null,
    };
  },
  async prefillWithTokenLogits(prompt, tokenIds, options) {
    assert.equal(prompt, '');
    assert.equal(options.useChatTemplate, false);
    assert.ok(Array.isArray(options.inputIds));
    assert.deepEqual(tokenIds, [scoringConfig.trueTokenId, scoringConfig.falseTokenId]);
    assert.ok(this.currentSeqLen > 0);
    this.prefixSelectedCalls++;
    const suffix = String.fromCharCode(...options.inputIds);
    const isPositive = positiveSnippets.some((snippet) => suffix.includes(snippet));
    const startSeqLen = this.currentSeqLen;
    this.currentSeqLen += options.inputIds.length;
    return {
      seqLen: this.currentSeqLen,
      prefixTokens: Array.from({ length: startSeqLen }, (_, i) => i + 1),
      tokens: options.inputIds,
      tokenIds,
      logits: Float32Array.from(isPositive ? [9, -1] : [-1, 9]),
      logitsByTokenId: {
        [scoringConfig.trueTokenId]: isPositive ? 9 : -1,
        [scoringConfig.falseTokenId]: isPositive ? -1 : 9,
      },
    };
  },
  async prefillWithLogits() {
    throw new Error('full logits should not be used when prefix scoring is available');
  },
};

const prefixRun = await runRerank(prefixPipeline, {
  inference: {
    rerank: {
      query: 'What is the capital of China?',
      documents: [
        'Gravity is a force that attracts two bodies towards each other.',
        'The capital of China is Beijing.',
      ],
    },
  },
});
assert.equal(prefixRun.documentCount, 2);
assert.equal(prefixRun.topDocument.index, 1);
assert.equal(prefixRun.ranking[0].scoringPath, 'prefix-selected-token-logits');
assert.equal(prefixRun.ranking[1].scoringPath, 'prefix-selected-token-logits');
assert.equal(prefixPipeline.resetCount, 1);
assert.equal(prefixPipeline.prefillKVOnlyCalls, 1);
assert.equal(prefixPipeline.prefixSelectedCalls, 2);
assert.ok(prefixPipeline.currentSeqLen > 0);
assert.ok(prefixPipeline.currentSeqLen < prefixRun.ranking[0].tokenCount);

const trueLogitScoringConfig = {
  ...scoringConfig,
  score: 'true_logit',
};
const trueLogitPipeline = {
  manifest: {
    inference: {
      supportsRerank: true,
      rerank: trueLogitScoringConfig,
    },
  },
  resetCount: 0,
  reset() {
    this.resetCount++;
  },
  async prefillWithLogits(_prompt, options) {
    assert.equal(options.useChatTemplate, false);
    const logits = new Float32Array(4);
    logits[trueLogitScoringConfig.trueTokenId] = 1.25;
    logits[trueLogitScoringConfig.falseTokenId] = 99;
    return { tokens: [1, 2], logits };
  },
};
const trueLogitResolvedConfig = resolveRerankScoringConfig(trueLogitPipeline);
assert.equal(trueLogitResolvedConfig.score, 'true_logit');
const trueLogitScore = await scoreRerankDocument(
  trueLogitPipeline,
  'q',
  'd',
  trueLogitResolvedConfig
);
assert.equal(trueLogitScore.score, 1.25);
assert.equal(trueLogitScore.trueLogit, 1.25);
assert.equal(trueLogitScore.falseLogit, 99);
assert.ok(trueLogitScore.probability > 0.77);
assert.equal(trueLogitPipeline.resetCount, 1);

const run = await runRerank(pipeline, {
  inference: {
    rerank: {
      query: 'What is the capital of China?',
      documents: [
        'Gravity is a force that attracts two bodies towards each other.',
        'The capital of China is Beijing.',
      ],
    },
  },
});
assert.equal(run.documentCount, 2);
assert.equal(run.topDocument.index, 1);
assert.equal(run.ranking[0].document, 'The capital of China is Beijing.');

const semantic = await runRerankSemanticChecks(pipeline, {
  rerankSemantic: {
    minPairAcc: 1,
    minScoreMargin: 0,
    cases: [
      {
        id: 'capital',
        query: 'What is the capital of China?',
        positive: 'The capital of China is Beijing.',
        negative: 'Gravity is a force that attracts two bodies towards each other.',
      },
    ],
  },
});
assert.equal(semantic.passed, true);
assert.equal(semantic.pairAcc, 1);
assert.equal(semantic.pairs[0].margin, 12);

assert.throws(
  () => formatRerankPrompt('q', 'd', { ...scoringConfig, inputTemplate: '{query} {document}' }),
  /inputTemplate is missing \{instruction\}/
);
assert.throws(
  () => resolveRerankScoringConfig({
    manifest: {
      inference: {
        rerank: {
          ...scoringConfig,
          score: 'implicit_default',
        },
      },
    },
  }),
  /Unsupported rerank score policy "implicit_default"/
);

console.log('rerank-scoring-details.test: ok');
