import { resolve } from 'path';
import { fileURLToPath } from 'url';

const CLI_DIR = resolve(fileURLToPath(import.meta.url), '..');
const PROJECT_ROOT = resolve(CLI_DIR, '..', '..');

export const TOOL_REGISTRY = {
  'attention-decode-grid': {
    id: 'attention-decode-grid',
    script: resolve(PROJECT_ROOT, 'tools', 'attention-decode-grid.js'),
    description: 'Attention decode microbench grid.',
    configKey: 'attentionDecodeGrid',
  },
  'kernel-path-grid': {
    id: 'kernel-path-grid',
    script: resolve(PROJECT_ROOT, 'tools', 'kernel-path-grid.js'),
    description: 'Benchmark kernel-path variants.',
    configKey: 'kernelPathGrid',
  },
  'validate-kernel-registry': {
    id: 'validate-kernel-registry',
    script: resolve(PROJECT_ROOT, 'tools', 'validate-kernel-registry.js'),
    description: 'Validate kernel registry WGSL references.',
    configKey: null,
  },
  'lint-wgsl-overrides': {
    id: 'lint-wgsl-overrides',
    script: resolve(PROJECT_ROOT, 'tools', 'lint-wgsl-overrides.js'),
    description: 'Lint WGSL override usage.',
    configKey: null,
  },
  'purge-opfs': {
    id: 'purge-opfs',
    script: resolve(PROJECT_ROOT, 'tools', 'purge-opfs.js'),
    description: 'Purge OPFS model cache entry.',
    configKey: null,
  },
  'update-manifest': {
    id: 'update-manifest',
    script: resolve(PROJECT_ROOT, 'tools', 'update-manifest.js'),
    description: 'Update manifest metadata fields.',
    configKey: 'updateManifest',
  },
  'generate-fixture': {
    id: 'generate-fixture',
    script: resolve(PROJECT_ROOT, 'tools', 'generate-fixture.js'),
    description: 'Generate a tiny fixture model.',
    configKey: 'generateFixture',
  },
  'test-query': {
    id: 'test-query',
    script: resolve(PROJECT_ROOT, 'tools', 'test-query.js'),
    description: 'Interactive inference query (browser harness).',
    configKey: 'testQuery',
  },
  'test-q4k-roundtrip': {
    id: 'test-q4k-roundtrip',
    script: resolve(PROJECT_ROOT, 'tools', 'test-q4k-roundtrip.js'),
    description: 'Inspect Q4K roundtrip behavior.',
    configKey: null,
  },
  'compare-quant': {
    id: 'compare-quant',
    script: resolve(PROJECT_ROOT, 'tools', 'compare-quant.js'),
    description: 'Compare converter vs reference quantization.',
    configKey: null,
  },
  'rdrr-lora-to-gguf': {
    id: 'rdrr-lora-to-gguf',
    script: resolve(PROJECT_ROOT, 'tools', 'rdrr-lora-to-gguf.js'),
    description: 'Write GGUF conversion plan for RDRR-LoRA.',
    configKey: 'rdrrLoraToGguf',
  },
  serve: {
    id: 'serve',
    script: resolve(PROJECT_ROOT, 'cli', 'commands', 'serve.js'),
    description: 'Convert (optional) and serve RDRR packs.',
    configKey: 'serve',
  },
  'diagnose-kernels': {
    id: 'diagnose-kernels',
    script: resolve(PROJECT_ROOT, 'src', 'debug', 'diagnose-kernels.js'),
    description: 'Diagnose kernel-path selection.',
    configKey: 'diagnoseKernels',
  },
};

export function listTools() {
  return Object.keys(TOOL_REGISTRY).sort();
}

export function getTool(id) {
  return TOOL_REGISTRY[id] ?? null;
}
