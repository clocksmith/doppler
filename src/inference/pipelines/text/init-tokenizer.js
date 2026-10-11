import { Tokenizer } from '../../tokenizer.js';
import { createGenerationOutputDecoder } from '../../generation-output.js';

/** @type {import('./init-tokenizer.js').initTokenizer} */
export async function initTokenizer(manifest, options = {}) {
  const { baseUrl, tokenizerHints, storageContext } = options;
  const tokenizer = new Tokenizer();
  await tokenizer.initialize(manifest, {
    baseUrl,
    tokenizerHints,
    loadTokenizerJson: typeof storageContext?.loadTokenizerJson === 'function'
      ? storageContext.loadTokenizerJson.bind(storageContext)
      : null,
    loadTokenizerModel: typeof storageContext?.loadTokenizerModel === 'function'
      ? storageContext.loadTokenizerModel.bind(storageContext)
      : null,
  });
  const stops = [...(Array.isArray(manifest.eos_token_id) ? manifest.eos_token_id : [manifest.eos_token_id]),
    tokenizer.getSpecialTokens?.()?.eos];
  const stopIds = [];
  for (const value of stops) {
    if (typeof value === 'number' && Number.isSafeInteger(value)) stopIds.push(value);
  }
  if (manifest.inference?.output?.reasoningOutput != null) {
    // Validate the actual backend's raw decoder before weight execution begins.
    createGenerationOutputDecoder(tokenizer, manifest.inference.output.reasoningOutput, stopIds);
  }
  return tokenizer;
}
