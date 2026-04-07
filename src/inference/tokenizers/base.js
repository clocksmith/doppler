

import { getRuntimeConfig } from '../../config/runtime.js';


export class BaseTokenizer {
  
  vocabSize;
  
  specialTokens;
  
  addBosToken;
  
  addEosToken;

  
  constructor(config = {}) {
    const runtimeDefaults = getRuntimeConfig().inference.tokenizer;
    const deferSpecialTokens = config.deferSpecialTokens ?? runtimeDefaults.deferSpecialTokens;
    const vocabSize = (config.vocabSize == null && deferSpecialTokens) ? 0 : config.vocabSize;
    if (vocabSize == null) {
      throw new Error('[Tokenizer] vocabSize is required.');
    }
    this.vocabSize = vocabSize;

    const specialTokens = config.specialTokens ?? {};
    
    const padToken = specialTokens.pad ?? config.padToken;
    const bosToken = specialTokens.bos ?? config.bosToken;
    const eosToken = specialTokens.eos ?? config.eosToken;
    const unkToken = specialTokens.unk ?? config.unkToken;

    if (!deferSpecialTokens && eosToken == null) {
      throw new Error('[Tokenizer] eosToken is required.');
    }
    if (!deferSpecialTokens && (config.addBosToken ?? runtimeDefaults.addBosToken) && bosToken == null) {
      throw new Error('[Tokenizer] bosToken is required when addBosToken is enabled.');
    }

    this.specialTokens = {
      pad: padToken,
      bos: bosToken,
      eos: eosToken,
      unk: unkToken,
    };
    this.addBosToken = config.addBosToken ?? runtimeDefaults.addBosToken;
    this.addEosToken = config.addEosToken ?? runtimeDefaults.addEosToken;
  }

  
  encode(text) {
    throw new Error('Abstract method not implemented');
  }

  
  decode(ids, skipSpecialTokens, trim) {
    throw new Error('Abstract method not implemented');
  }

  
  getVocabSize() {
    return this.vocabSize;
  }

  getHotTokenIds(limit) {
    void limit;
    return null;
  }

  
  isSpecialToken(tokenId) {
    return Object.values(this.specialTokens).includes(tokenId);
  }
}
