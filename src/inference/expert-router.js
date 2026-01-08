/**
 * Expert router for multi-model execution.
 *
 * @module inference/expert-router
 */

/**
 * @typedef {Object} ExpertProfile
 * @property {string} id
 * @property {number[]} [embedding]
 * @property {Record<string, unknown>} [metadata]
 */

export class ExpertRouter {
  /** @type {Map<string, ExpertProfile>} */
  experts;

  constructor() {
    this.experts = new Map();
  }

  /**
   * @param {ExpertProfile} profile
   * @returns {void}
   */
  registerExpert(profile) {
    this.experts.set(profile.id, profile);
  }

  /**
   * @param {string} id
   * @returns {void}
   */
  removeExpert(id) {
    this.experts.delete(id);
  }

  /**
   * @returns {ExpertProfile[]}
   */
  listExperts() {
    return Array.from(this.experts.values());
  }

  /**
   * @param {number[]} embedding
   * @param {number} [topK=1]
   * @returns {ExpertProfile[]}
   */
  selectByEmbedding(embedding, topK = 1) {
    /** @type {{ expert: ExpertProfile; score: number }[]} */
    const scored = [];
    for (const expert of this.experts.values()) {
      if (!expert.embedding) continue;
      const score = this.cosineSimilarity(embedding, expert.embedding);
      scored.push({ expert, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map((item) => item.expert);
  }

  /**
   * @param {number[]} a
   * @param {number[]} b
   * @returns {number}
   */
  cosineSimilarity(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
  }
}
