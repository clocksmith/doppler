



export class ExpertRouter {
  
  experts;

  constructor() {
    this.experts = new Map();
  }

  
  registerExpert(profile) {
    this.experts.set(profile.id, profile);
  }

  
  removeExpert(id) {
    this.experts.delete(id);
  }

  
  listExperts() {
    return Array.from(this.experts.values());
  }

  
  selectByEmbedding(embedding, topK = 1) {
    
    const scored = [];
    for (const expert of this.experts.values()) {
      if (!expert.embedding) continue;
      const score = this.cosineSimilarity(embedding, expert.embedding);
      scored.push({ expert, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map((item) => item.expert);
  }

  
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
