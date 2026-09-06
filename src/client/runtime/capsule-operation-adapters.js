import { requireGenerationOptions } from './session-controller.js';

const text = (value) => typeof value === 'string' && value.trim().length > 0;
const texts = (value) => Array.isArray(value) && value.length > 0 && value.every(text);
const requireValue = (condition, message) => { if (!condition) throw new Error(message); };

export function createCapsuleOperationAdapters({ program, generate, rerank, embed, encodeSequence }) {
  return {
    generate: {
      validate({ input, options }) {
        requireValue(Object.hasOwn(input, 'prompt') !== Object.hasOwn(input, 'promptTokens'), 'generate requires exactly one prompt or promptTokens input.');
        if (Object.hasOwn(input, 'prompt')) requireValue(text(input.prompt), 'generate requires a non-empty prompt.');
        if (Object.hasOwn(input, 'promptTokens')) requireValue(Array.isArray(input.promptTokens) && input.promptTokens.length > 0 && input.promptTokens.every((id) => Number.isSafeInteger(id) && id >= 0), 'Invalid prompt token IDs.');
        requireGenerationOptions(options);
        if (options.stopSequences !== undefined) requireValue(Array.isArray(options.stopSequences) && options.stopSequences.every(text), 'Invalid stop sequences.');
        if (options.suppressTokenIds !== undefined) requireValue(Array.isArray(options.suppressTokenIds) && options.suppressTokenIds.every((id) => Number.isSafeInteger(id) && id >= 0), 'Invalid suppressed token IDs.');
      },
      async *execute({ input, options }, signal) {
        const tokenIds = [];
        for await (const tokenId of generate({ ...input, ...options, signal })) {
          requireValue(Number.isSafeInteger(tokenId) && tokenId >= 0, 'Invalid generated token ID.');
          tokenIds.push(tokenId);
          const output = { text: program.decodeTokens(tokenIds), tokenIds: [...tokenIds] };
          yield { delta: { tokenId }, output };
        }
        return { text: program.decodeTokens(tokenIds), tokenIds };
      },
    },
    embed: {
      validate({ input }) {
        requireValue(texts(input.texts), 'embed requires a non-empty texts array.');
        requireValue(input.application && typeof input.application === 'object', 'embed requires its signed application binding.');
      },
      async *execute({ input, options }, signal) {
        const embeddings = [];
        for (const value of input.texts) {
          signal.throwIfAborted();
          const result = await embed({ application: input.application, text: value, options: { ...options, signal } });
          requireValue(Array.isArray(result?.embedding) || ArrayBuffer.isView(result?.embedding), 'Capsule embed must return an embedding vector.');
          requireValue(result.embedding.length > 0, 'Capsule embed returned an empty vector.');
          requireValue(!embeddings.length || result.embedding.length === embeddings[0].embedding.length, 'Capsule embed returned inconsistent vector dimensions.');
          embeddings.push(result);
          // A completed batch item is partial job output, never acceptance.
          yield { delta: { itemIndex: embeddings.length - 1 }, output: { embeddings: [...embeddings] } };
        }
        return { embeddings };
      },
    },
    rerank: {
      validate({ input }) {
        requireValue(text(input.query) && texts(input.documents), 'rerank requires query and non-empty documents.');
        requireValue(input.application && typeof input.application === 'object', 'rerank requires its signed application binding.');
      },
      async *execute({ input, options }, signal) {
        return await rerank({ ...input, options: { ...options, signal } });
      },
    },
    encodeSequence: {
      validate({ input, options }) {
        requireValue(text(input.sequence), 'encodeSequence requires a sequence.');
        requireValue(typeof options.includeLogits === 'boolean' && typeof options.includeTokenEmbeddings === 'boolean', 'encodeSequence requires explicit output flags.');
      },
      async *execute({ input, options, assignment }, signal) {
        return await encodeSequence(input.sequence, { ...options, assignment, signal });
      },
    },
  };
}
