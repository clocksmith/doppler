/**
 * Adapts a raw Doppler pipeline into the model handle shape that
 * createDopplerProvider() expects, without triggering a load.
 *
 * @param {Object} pipeline - A loaded Doppler pipeline (has generate, manifest, etc.)
 * @param {{ modelId?: string, manifest?: Object, deviceInfo?: Object }} [resolved]
 * @returns {{ loaded: boolean, modelId: string, manifest: Object|null, deviceInfo: Object|null, generateText(prompt: unknown, opts?: Object): Promise<string>, unload(): Promise<void> }}
 */
export function wrapPipelineAsHandle(pipeline, resolved = {}) {
  if (!pipeline || typeof pipeline.generate !== 'function') {
    throw new Error('wrapPipelineAsHandle requires a loaded pipeline with a generate() method.');
  }

  async function collectText(iterable) {
    let result = '';
    for await (const chunk of iterable) {
      if (typeof chunk === 'string') {
        result += chunk;
      } else if (chunk && typeof chunk.text === 'string') {
        result += chunk.text;
      }
    }
    return result;
  }

  return {
    get loaded() {
      return pipeline.isLoaded === true;
    },
    get modelId() {
      return String(resolved.modelId || pipeline.manifest?.meta?.modelId || '');
    },
    get manifest() {
      return pipeline.manifest || resolved.manifest || null;
    },
    get deviceInfo() {
      return resolved.deviceInfo || null;
    },
    get supportsEmbedding() {
      const manifest = pipeline.manifest || resolved.manifest || null;
      return manifest?.modelType === 'embedding'
        || manifest?.inference?.supportsEmbedding === true;
    },
    get supportsTranscription() {
      const manifest = pipeline.manifest || resolved.manifest || null;
      return manifest?.inference?.supportsTranscription === true
        && pipeline.audioCapable === true;
    },
    get supportsVision() {
      const manifest = pipeline.manifest || resolved.manifest || null;
      return manifest?.inference?.supportsVision === true
        && pipeline.visionCapable === true;
    },
    async generateText(prompt, opts = {}) {
      return collectText(pipeline.generate(prompt, opts));
    },
    async embed(prompt, options = {}) {
      return pipeline.embed(prompt, options);
    },
    async embedBatch(prompts, options = {}) {
      return pipeline.embedBatch(prompts, options);
    },
    async embedImage(args = {}) {
      return pipeline.embedImage(args);
    },
    async embedAudio(args = {}) {
      return pipeline.embedAudio(args);
    },
    async transcribeImage(args = {}) {
      return pipeline.transcribeImage(args);
    },
    async transcribeAudio(args = {}) {
      return pipeline.transcribeAudio(args);
    },
    async transcribeVideo(args = {}) {
      return pipeline.transcribeVideo(args);
    },
    async unload() {
      if (typeof pipeline.unload === 'function') {
        await pipeline.unload();
      }
    },
  };
}
