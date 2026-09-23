// Application retrieval policy stays outside the inference runtime.
export function createDocumentSearch({ embedding, reranker, embeddingApplication, rerankerApplication,
  embeddingIdentity, dimension, candidateCount, queryPrefix, documentPrefix }) {
  if (!Number.isSafeInteger(dimension) || dimension < 1 || !Number.isSafeInteger(candidateCount) || candidateCount < 1
    || typeof queryPrefix !== 'string' || typeof documentPrefix !== 'string' || !embeddingIdentity) {
    throw new Error('Explicit search geometry, candidate policy, prefixes and embedding identity required.');
  }
  const vector = async (text, signal) => {
    const result = await embedding.embed({ application: embeddingApplication, text, options: { signal } });
    const values = Array.from(result.embedding);
    if (values.length !== dimension || !values.every(Number.isFinite)) throw new Error('Invalid model embedding.');
    return values;
  };
  const identity = JSON.stringify(embeddingIdentity);
  function assertIndex(index) {
    if (index?.schema !== 'doppler.document-search-index/v1' || index.embeddingIdentity !== identity
      || index.dimension !== dimension || index.documentPrefix !== documentPrefix || !Array.isArray(index.documents)) {
      const error = new Error('Search index is incompatible with this embedding release; rebuild it from retained documents.');
      error.code = 'DOCUMENT_SEARCH_INDEX_INCOMPATIBLE';
      throw error;
    }
    const ids = new Set();
    for (const document of index.documents) {
      if (typeof document.id !== 'string' || !document.id || ids.has(document.id)
        || typeof document.text !== 'string' || typeof document.title !== 'string'
        || !Array.isArray(document.vector) || document.vector.length !== dimension || !document.vector.every(Number.isFinite)) {
        throw new Error('Invalid retained document index.');
      }
      ids.add(document.id);
    }
  }
  async function digest(text) {
    const bytes = new TextEncoder().encode(text);
    const hash = await crypto.subtle.digest('SHA-256', bytes);
    return 'sha256:' + Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2, '0')).join('');
  }
  return {
    assertIndex,
    async indexDocuments(documents, priorIndexOrOptions = null, optionsOrEmpty = {}) {
      let priorIndex = priorIndexOrOptions;
      let options = optionsOrEmpty;
      if (priorIndexOrOptions && !('documents' in priorIndexOrOptions) && !('schema' in priorIndexOrOptions)) {
        options = priorIndexOrOptions;
        priorIndex = null;
      }
      if (priorIndex != null) {
        if (priorIndex?.schema !== 'doppler.document-search-index/v1' || priorIndex.embeddingIdentity !== identity
          || priorIndex.dimension !== dimension || priorIndex.documentPrefix !== documentPrefix || !Array.isArray(priorIndex.documents)) {
          const error = new Error('Search index is incompatible with this embedding release; rebuild it from retained documents.');
          error.code = 'DOCUMENT_SEARCH_INDEX_INCOMPATIBLE';
          throw error;
        }
        assertIndex(priorIndex);
      }
      const priorDocs = new Map();
      if (priorIndex?.documents) {
        for (const doc of priorIndex.documents) {
          if (doc?.id) priorDocs.set(doc.id, doc);
        }
      }
      const index = { schema: 'doppler.document-search-index/v1', embeddingIdentity: identity, dimension, documentPrefix, documents: [] };
      for (const document of documents) {
        options.signal?.throwIfAborted();
        if (!['text/plain', 'text/markdown'].includes(document.mediaType)) throw new Error('Only text and Markdown documents are supported.');
        if (typeof document.text !== 'string' || !document.text.trim()) throw new Error('Document text is required.');
        if (typeof document.id !== 'string' || !document.id.trim()) throw new Error('Document id is required.');
        if (typeof document.title !== 'string') throw new Error('Document title is required.');
        const contentHash = await digest(documentPrefix + document.text);
        const prior = priorDocs.get(document.id);
        let docVector;
        if (prior && prior.contentHash === contentHash && Array.isArray(prior.vector)
          && prior.vector.length === dimension && prior.vector.every(Number.isFinite)) {
          docVector = prior.vector;
        } else {
          options.signal?.throwIfAborted();
          docVector = await vector(documentPrefix + document.text, options.signal);
        }
        index.documents.push({ ...document, contentHash, vector: docVector });
      }
      options.signal?.throwIfAborted();
      assertIndex(index);
      return index;
    },
    async search(index, query, options = {}) {
      assertIndex(index);
      if (typeof query !== 'string' || !query.trim()) throw new Error('A search query is required.');
      options.signal?.throwIfAborted();
      const queryVector = await vector(queryPrefix + query, options.signal);
      options.signal?.throwIfAborted();
      const candidates = index.documents.map((document, position) => {
        let dot = 0; let left = 0; let right = 0;
        for (let i = 0; i < dimension; i++) {
          dot += queryVector[i] * document.vector[i];
          left += queryVector[i] * queryVector[i]; right += document.vector[i] * document.vector[i];
        }
        if (!(left > 0 && right > 0)) throw new Error('Zero-length retrieval vector.');
        return { document, position, similarity: dot / Math.sqrt(left * right) };
      }).sort((a, b) => b.similarity - a.similarity || a.position - b.position).slice(0, candidateCount);
      if (!candidates.length) return { query, candidates: [], results: [], receipt: null };
      const receipt = await reranker.rerank({ application: rerankerApplication, query,
        documents: candidates.map(candidate => candidate.document.text), options });
      const results = receipt.evidence.scores.map(score => ({ ...candidates[score.index], rerankScore: score.score }))
        .sort((a, b) => b.rerankScore - a.rerankScore || a.position - b.position);
      return { query, candidates, results, receipt };
    },
  };
}
