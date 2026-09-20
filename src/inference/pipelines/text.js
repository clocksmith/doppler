import { InferencePipeline, EmbeddingPipeline } from './text/pipeline.js';
export * from './text/pipeline.js';
import { createInitializedPipeline } from './factory.js';
import { registerPipeline, getPipelineFactory } from './registry.js';
import { selectRuleValue } from '../../rules/rule-registry.js';

/** @param {import('../../config/schema/index.js').ManifestSchema} manifest
 * @param {Record<string, unknown>} contexts */
async function createTransformerPipeline(manifest, contexts = {}) {
  return createInitializedPipeline(InferencePipeline, manifest, contexts);
}

registerPipeline('transformer', createTransformerPipeline);
registerPipeline('gemma4', createTransformerPipeline);

/** @param {import('../../config/schema/index.js').ManifestSchema} manifest
 * @param {Record<string, unknown>} contexts */
async function createEmbeddingPipeline(manifest, contexts = {}) {
  return createInitializedPipeline(EmbeddingPipeline, manifest, contexts);
}

registerPipeline('embedding', createEmbeddingPipeline);

/** @param {string} modelType */
function resolveLazyPipelineModules(modelType) {
  const modules = selectRuleValue('inference', 'config', 'pipelineModules', {
    modelType,
    modelTypeLower: String(modelType).toLowerCase(),
  });
  if (!Array.isArray(modules)) return [];
  return modules.filter((entry) => typeof entry === 'string' && entry.length > 0);
}

/** @param {import('../../config/schema/index.js').ManifestSchema} manifest
 * @param {Record<string, unknown>} contexts */
export async function createPipeline(manifest, contexts = {}) {
  const modelType = manifest?.modelType;
  if (typeof modelType !== 'string' || modelType.length === 0) {
    throw new Error('Manifest is missing modelType. Re-convert the model with modelType set.');
  }
  let factory = getPipelineFactory(modelType);

  if (!factory) {
    for (const modulePath of resolveLazyPipelineModules(modelType)) {
      await import(modulePath);
    }
    factory = getPipelineFactory(modelType);
  }

  if (!factory) {
    throw new Error(`No pipeline registered for modelType "${modelType}".`);
  }

  return factory(manifest, contexts);
}
