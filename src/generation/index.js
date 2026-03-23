import { createPipeline as _createPipeline } from '../inference/pipelines/text.js';
import { log } from '../debug/index.js';

export { InferencePipeline, EmbeddingPipeline } from '../inference/pipelines/text.js';
export { parseModelConfig, parseModelConfigFromManifest } from '../inference/pipelines/text/config.js';
export { loadWeights, initTokenizer, isStopToken } from '../inference/pipelines/text/init.js';
export { initTokenizerFromManifest } from '../inference/pipelines/text/model-load.js';
export {
  StructuredJsonHeadPipeline,
  isStructuredJsonHeadModelType,
  createStructuredJsonHeadPipeline,
  DreamStructuredPipeline,
  isDreamStructuredModelType,
  createDreamStructuredPipeline,
} from '../inference/pipelines/structured/json-head-pipeline.js';

/** Required methods that any pipeline returned from createPipeline must implement. */
const REQUIRED_PIPELINE_METHODS = ['generate', 'loadModel', 'reset', 'unload'];

export async function createPipeline(manifest, contexts = {}) {
  const pipeline = await _createPipeline(manifest, contexts);
  for (const method of REQUIRED_PIPELINE_METHODS) {
    if (typeof pipeline[method] !== 'function') {
      log.warn(
        'Generation',
        `Created pipeline for modelType "${manifest?.modelType}" is missing required method "${method}". ` +
        'This may cause failures during generation.'
      );
    }
  }
  return pipeline;
}
