import { resolveGenerationOptions, validateGenerationInput } from '../../config/generation-contract.js';

const text = (value) => typeof value === 'string' && value.trim().length > 0;
const texts = (value) => Array.isArray(value) && value.length > 0 && value.every(text);
const requireValue = (condition, message) => { if (!condition) throw new Error(message); };

export function createCapsuleOperationAdapters({ program, generate, rerank, embed, encodeSequence }) {
  return {
    generate: {
      validate({ input, options }) {
        validateGenerationInput(input);
        resolveGenerationOptions(options);
      },
      async *execute({ input, options }, signal) {
        const tokenIds = [];
        const iterator = generate({ ...input, ...options, signal });
        try {
          while (true) {
            const next = await iterator.next();
            if (next.done) return { text: program.decodeTokens(tokenIds), tokenIds, ...next.value };
            const tokenId = next.value;
            requireValue(Number.isSafeInteger(tokenId) && tokenId >= 0, 'Invalid generated token ID.');
            tokenIds.push(tokenId);
            const output = { text: program.decodeTokens(tokenIds), tokenIds: [...tokenIds] };
            yield { delta: { tokenId }, output };
          }
        } finally { await iterator.return?.(); }
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
