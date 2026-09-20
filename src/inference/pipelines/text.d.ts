export * from './text/pipeline.js';
import type { InferencePipeline, EmbeddingPipeline, PipelineContexts } from './text/pipeline.js';
import type { Manifest } from './text/config.js';
import type { DiffusionPipeline } from '../../experimental/diffusion/pipeline.js';
import type { EnergyPipeline } from '../../experimental/energy/pipeline.js';
import type { StructuredJsonHeadPipeline } from './structured/json-head-pipeline.js';
import type { EnergyRowHeadPipeline } from './energy-head/row-head-pipeline.js';

export declare function createPipeline(
  manifest: Manifest,
  contexts?: PipelineContexts
): Promise<
  InferencePipeline |
  EmbeddingPipeline |
  DiffusionPipeline |
  EnergyPipeline |
  StructuredJsonHeadPipeline |
  EnergyRowHeadPipeline
>;
