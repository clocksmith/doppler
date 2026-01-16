

import { getRuntimeConfig } from '../../config/runtime.js';


export class BaseTokenizer {
  
  vocabSize;
  
  specialTokens;
  
  addBosToken;
  
  addEosToken;

  
  constructor(config = {}) {
    const runtimeDefaults = getRuntimeConfig().inference.tokenizer;
    this.vocabSize = config.vocabSize || 32000;
    this.specialTokens = {
      pad: config.padToken ?? 0,
      bos: config.bosToken ?? 1,
      eos: config.eosToken ?? 2,
      unk: config.unkToken ?? 0,
      ...config.specialTokens
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

  
  isSpecialToken(tokenId) {
    return Object.values(this.specialTokens).includes(tokenId);
  }
}
