import { PipelineGenerator } from '../inference/pipelines/text/generator.js';

export const DOPPLER_PROGRAM_BUNDLE_HOST_SUBSET = 'doppler-webgpu-host/v1';

export function createTextGenerationProgram(state) {
  return new PipelineGenerator(state);
}

export function describeTextGenerationProgram() {
  return {
    schema: 'doppler.host-js/v1',
    jsSubset: DOPPLER_PROGRAM_BUNDLE_HOST_SUBSET,
    entrypoint: {
      id: 'text-generation',
      module: 'src/tooling/program-bundle-host.js',
      export: 'createTextGenerationProgram',
      role: 'model-orchestration',
    },
  };
}
