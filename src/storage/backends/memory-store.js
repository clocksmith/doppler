function ensureModel(models, modelId) {
  if (!models.has(modelId)) {
    models.set(modelId, { files: new Map(), meta: new Map() });
  }
  return models.get(modelId);
}

export function createMemoryStore(config) {
  const { maxBytes } = config;
  const models = new Map();
  let currentModelId = null;
  let totalBytes = 0;

  async function init() {}

  async function openModel(modelId, options = {}) {
    const create = options.create !== false;
    if (!create && !models.has(modelId)) {
      throw new Error('Model not found');
    }
    currentModelId = modelId;
    if (create) {
      ensureModel(models, modelId);
    }
    return null;
  }

  function getCurrentModelId() {
    return currentModelId;
  }

  function requireModel() {
    if (!currentModelId) {
      throw new Error('No model open. Call openModelStore first.');
    }
  }

  function adjustBytes(delta) {
    totalBytes += delta;
    if (totalBytes > maxBytes) {
      totalBytes -= delta;
      throw new Error(`Memory store exceeded maxBytes (${maxBytes})`);
    }
  }

  async function readFile(filename) {
    requireModel();
    const model = ensureModel(models, currentModelId);
    const entry = model.files.get(filename);
    if (!entry) {
      throw new Error(`File not found: ${filename}`);
    }
    return entry.buffer.slice(entry.byteOffset, entry.byteOffset + entry.byteLength);
  }

  async function readText(filename) {
    try {
      const buffer = await readFile(filename);
      return new TextDecoder().decode(buffer);
    } catch (error) {
      if (error.message?.includes('File not found')) {
        return null;
      }
      throw error;
    }
  }

  async function writeFile(filename, data) {
    requireModel();
    const model = ensureModel(models, currentModelId);
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
    const prev = model.files.get(filename);
    if (prev) {
      adjustBytes(-prev.byteLength);
    }
    adjustBytes(bytes.byteLength);
    model.files.set(filename, bytes.slice(0));
  }

  async function createWriteStream(filename) {
    requireModel();
    const model = ensureModel(models, currentModelId);
    let chunks = [];
    let total = 0;
    return {
      write: async (chunk) => {
        const bytes = chunk instanceof ArrayBuffer ? new Uint8Array(chunk) : chunk;
        chunks.push(bytes.slice(0));
        total += bytes.byteLength;
      },
      close: async () => {
        const combined = new Uint8Array(total);
        let offset = 0;
        for (const chunk of chunks) {
          combined.set(chunk, offset);
          offset += chunk.byteLength;
        }
        const prev = model.files.get(filename);
        if (prev) {
          adjustBytes(-prev.byteLength);
        }
        adjustBytes(combined.byteLength);
        model.files.set(filename, combined);
        chunks = [];
      },
      abort: async () => {
        chunks = [];
      },
    };
  }

  async function deleteFile(filename) {
    requireModel();
    const model = ensureModel(models, currentModelId);
    const prev = model.files.get(filename);
    if (prev) {
      adjustBytes(-prev.byteLength);
      model.files.delete(filename);
    }
    return true;
  }

  async function listFiles() {
    requireModel();
    const model = ensureModel(models, currentModelId);
    return Array.from(model.files.keys());
  }

  async function listModels() {
    return Array.from(models.keys());
  }

  async function deleteModel(modelId) {
    const model = models.get(modelId);
    if (model) {
      for (const value of model.files.values()) {
        adjustBytes(-value.byteLength);
      }
      models.delete(modelId);
    }
    if (currentModelId === modelId) {
      currentModelId = null;
    }
    return true;
  }

  async function writeManifest(text) {
    requireModel();
    const model = ensureModel(models, currentModelId);
    model.meta.set('manifest', text);
  }

  async function readManifest() {
    requireModel();
    const model = ensureModel(models, currentModelId);
    return model.meta.get('manifest') ?? null;
  }

  async function writeTokenizer(text) {
    requireModel();
    const model = ensureModel(models, currentModelId);
    model.meta.set('tokenizer', text);
  }

  async function readTokenizer() {
    requireModel();
    const model = ensureModel(models, currentModelId);
    return model.meta.get('tokenizer') ?? null;
  }

  async function cleanup() {
    models.clear();
    currentModelId = null;
    totalBytes = 0;
  }

  return {
    init,
    openModel,
    getCurrentModelId,
    readFile,
    readText,
    writeFile,
    createWriteStream,
    deleteFile,
    listFiles,
    listModels,
    deleteModel,
    writeManifest,
    readManifest,
    writeTokenizer,
    readTokenizer,
    cleanup,
  };
}
