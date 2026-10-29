// Structured-generation helpers (grammar masks, JSON-head pipeline, etc.).
// Narrow entry point so consumers that only need mask construction don't
// pull the full inference pipeline into their bundle.

export { StructuredJsonHeadPipeline, isStructuredJsonHeadModelType, createStructuredJsonHeadPipeline, DreamStructuredPipeline, isDreamStructuredModelType, createDreamStructuredPipeline } from '../inference/pipelines/structured/json-head-pipeline.js';
export { createJsonGrammarMask } from '../inference/pipelines/structured/json-grammar-mask.js';
