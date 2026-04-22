import type { PipelineGenerator } from '../inference/pipelines/text/generator.js';

export declare const DOPPLER_PROGRAM_BUNDLE_HOST_SUBSET: 'doppler-webgpu-host/v1';

export declare function createTextGenerationProgram(state: unknown): PipelineGenerator;

export declare function describeTextGenerationProgram(): {
  schema: 'doppler.host-js/v1';
  jsSubset: typeof DOPPLER_PROGRAM_BUNDLE_HOST_SUBSET;
  entrypoint: {
    id: 'text-generation';
    module: 'src/tooling/program-bundle-host.js';
    export: 'createTextGenerationProgram';
    role: 'model-orchestration';
  };
};
