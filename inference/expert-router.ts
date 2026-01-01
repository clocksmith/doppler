/**
 * Expert router for multi-model execution.
 *
 * @module inference/expert-router
 */

export interface ExpertProfile {
  id: string;
  embedding?: number[];
  metadata?: Record<string, unknown>;
}

export class ExpertRouter {
  private experts: Map<string, ExpertProfile>;

  constructor() {
    this.experts = new Map();
  }

  registerExpert(profile: ExpertProfile): void {
    this.experts.set(profile.id, profile);
  }

  removeExpert(id: string): void {
    this.experts.delete(id);
  }

  listExperts(): ExpertProfile[] {
    return Array.from(this.experts.values());
  }

  selectByEmbedding(embedding: number[], topK: number = 1): ExpertProfile[] {
    const scored = [];
    for (const expert of this.experts.values()) {
      if (!expert.embedding) continue;
      const score = this.cosineSimilarity(embedding, expert.embedding);
      scored.push({ expert, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map((item) => item.expert);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
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
