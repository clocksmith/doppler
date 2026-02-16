export async function createInitializedPipeline(PipelineClass, manifest, contexts = {}) {
  const pipeline = new PipelineClass();
  await pipeline.initialize(contexts);
  await pipeline.loadModel(manifest);
  return pipeline;
}
