// Structured-generation helpers (grammar masks, JSON-head pipeline, etc.).
// Narrow entry point so consumers that only need mask construction don't
// pull the full inference pipeline into their bundle.

export * from '../inference/pipelines/structured/index.js';
