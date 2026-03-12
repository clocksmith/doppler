export async function parseTransformerModel(adapter) {
  const {
    readJson,
    fileExists,
    loadSingleSafetensors,
    loadShardedSafetensors,
  } = adapter;

  const config = await readJson('config.json', 'config.json');
  const generationConfig = await fileExists('generation_config.json')
    ? await readJson('generation_config.json', 'generation_config.json')
    : null;
  const architectureHint = config.architectures?.[0] ?? config.model_type ?? '';

  let tensors = null;
  if (await fileExists('model.safetensors.index.json')) {
    const indexJson = await readJson('model.safetensors.index.json', 'model.safetensors.index.json');
    tensors = await loadShardedSafetensors(indexJson);
  } else {
    tensors = await loadSingleSafetensors('model.safetensors');
  }

  return {
    config,
    generationConfig,
    tensors,
    architectureHint,
  };
}
