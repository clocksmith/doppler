import { registerPipeline } from '../pipeline/registry.js';

export async function createDiffusionPipeline() {
  throw new Error('Diffusion pipeline is not implemented yet.');
}

registerPipeline('diffusion', createDiffusionPipeline);
