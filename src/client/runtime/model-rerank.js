import { resolveRerankScoringConfig, scoreRerankDocument } from '../../inference/rerank.js';

export async function collectModelRerankScores(pipeline, query, documents, options) {
  if (pipeline.manifest?.inference?.supportsRerank !== true) {
    throw new Error('Loaded Doppler manifest does not declare rerank support.');
  }
  const normalizedQuery = String(query || '').trim();
  if (!normalizedQuery) throw new Error('Doppler rerankWithEvidence requires a non-empty query.');
  if (!Array.isArray(documents) || documents.length === 0) {
    throw new Error('Doppler rerankWithEvidence requires a non-empty documents array.');
  }
  const normalizedDocuments = documents.map((document, index) => {
    const value = String(document || '').trim();
    if (!value) throw new Error(`Doppler rerank document ${index} must be non-empty.`);
    return value;
  });
  const scoringConfig = resolveRerankScoringConfig(pipeline);
  const scores = [];
  for (let index = 0; index < normalizedDocuments.length; index += 1) {
    const scored = await scoreRerankDocument(pipeline, normalizedQuery, normalizedDocuments[index],
      scoringConfig, { benchmark: options.benchmark === true });
    scores.push({ index, document: normalizedDocuments[index], score: scored.score,
      probability: scored.probability, trueLogit: scored.trueLogit, falseLogit: scored.falseLogit,
      tokenCount: scored.tokenCount, tokenIds: scored.tokenIds, scoringPath: scored.scoringPath });
  }
  const ranking = [...scores].sort((left, right) => right.score - left.score || left.index - right.index)
    .map((entry, index) => ({ rank: index + 1, ...entry }));
  return { normalizedQuery, normalizedDocuments, scores, ranking };
}
