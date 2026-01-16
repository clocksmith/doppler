

import { writeFile } from 'fs/promises';
import { join } from 'path';
import { log } from '../../debug/index.js';
import { selectRuleValue } from '../../rules/rule-registry.js';


export class TokenizerWriter {
  #outputDir;

  constructor(outputDir) {
    this.#outputDir = outputDir;
  }

  
  async writeTokenizer(tokenizer) {
    if (!tokenizer.tokens || tokenizer.tokens.length === 0) {
      log.warn('TokenizerWriter', 'No vocab tokens found, skipping tokenizer bundling');
      return {
        model: tokenizer.model,
        bosTokenId: tokenizer.bosTokenId,
        eosTokenId: tokenizer.eosTokenId,
        padTokenId: tokenizer.padTokenId,
        unkTokenId: tokenizer.unkTokenId,
        addBosToken: tokenizer.addBosToken,
        addEosToken: tokenizer.addEosToken,
      };
    }

    const vocab = {};
    for (let i = 0; i < tokenizer.tokens.length; i++) {
      vocab[tokenizer.tokens[i]] = i;
    }

    const hasMerges = tokenizer.merges && tokenizer.merges.length > 0;
    const hasScores = tokenizer.scores && tokenizer.scores.length > 0;
    const type = selectRuleValue('converter', 'tokenizer', 'type', { hasMerges, hasScores });

    const tokenizerJson = {
      type,
      model: tokenizer.model,
      vocab,
      vocabSize: tokenizer.tokens.length,
      merges: hasMerges ? tokenizer.merges : null,
      scores: hasScores ? tokenizer.scores : null,
      tokenTypes: tokenizer.tokenTypes ?? null,
      specialTokens: {
        bos: tokenizer.bosTokenId,
        eos: tokenizer.eosTokenId,
        pad: tokenizer.padTokenId,
        unk: tokenizer.unkTokenId,
        sep: tokenizer.sepTokenId,
        cls: tokenizer.clsTokenId,
        mask: tokenizer.maskTokenId,
      },
      addBosToken: tokenizer.addBosToken ?? true,
      addEosToken: tokenizer.addEosToken ?? false,
      addSpacePrefix: tokenizer.addSpacePrefix ?? true,
    };

    const tokenizerPath = join(this.#outputDir, 'tokenizer.json');
    await writeFile(tokenizerPath, JSON.stringify(tokenizerJson));

    log.verbose('TokenizerWriter', `Wrote tokenizer.json (${tokenizer.tokens.length} tokens, type: ${type})`);

    return {
      type: 'bundled',
      file: 'tokenizer.json',
      vocabSize: tokenizer.tokens.length,
      tokenizerType: type,
    };
  }

  
  async writeHuggingFaceTokenizer(tokenizerJson) {
    if (!tokenizerJson || !tokenizerJson.model) {
      log.warn('TokenizerWriter', 'Invalid HuggingFace tokenizer.json, skipping');
      return null;
    }

    const tokenizerPath = join(this.#outputDir, 'tokenizer.json');
    await writeFile(tokenizerPath, JSON.stringify(tokenizerJson));

    const model = tokenizerJson.model;
    let vocabSize;
    if (Array.isArray(model.vocab)) {
      vocabSize = model.vocab.length;
    } else {
      vocabSize = Object.keys(model.vocab || {}).length;
    }
    const type = model.type?.toLowerCase() || 'bpe';

    log.verbose('TokenizerWriter', `Wrote HuggingFace tokenizer.json (${vocabSize} tokens, type: ${type})`);

    return {
      type: 'huggingface',
      file: 'tokenizer.json',
      vocabSize,
      tokenizerType: type,
    };
  }
}
