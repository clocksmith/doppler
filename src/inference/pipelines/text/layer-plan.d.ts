/**
 * Layer pipeline plan compiler.
 *
 * Converts JSON-configured step lists into normalized, validated plans
 * for execution inside processLayer.
 *
 * @module inference/pipelines/text/layer-plan
 */

import type {
  LayerPipelineSchema,
  LayerPipelineOp,
  LayerPipelineNormWeight,
  ProbeStage,
} from '../../../config/schema/index.js';

export interface CompiledLayerPipelineStep {
  op: LayerPipelineOp;
  src: string;
  dst: string;
  name?: string;
  weight?: LayerPipelineNormWeight;
  residual?: string | null;
  a?: string;
  b?: string;
  variant?: 'auto' | 'dense' | 'moe';
  skipInputNorm?: boolean;
  probeStage?: ProbeStage;
}

export interface LayerPipelineOverride {
  layers: number[];
  steps: CompiledLayerPipelineStep[];
}

export interface CompiledLayerPipeline {
  steps: CompiledLayerPipelineStep[];
  overrides: LayerPipelineOverride[];
  source: 'model' | 'runtime';
}

export function compileLayerPipeline(
  plan: LayerPipelineSchema,
  numLayers: number
): Omit<CompiledLayerPipeline, 'source'>;

export function resolveLayerPipeline(
  modelPlan: LayerPipelineSchema | null | undefined,
  runtimePlan: LayerPipelineSchema | null | undefined,
  numLayers: number
): CompiledLayerPipeline | null;

export function getLayerPlanSteps(plan: CompiledLayerPipeline, layerIdx: number): CompiledLayerPipelineStep[];
