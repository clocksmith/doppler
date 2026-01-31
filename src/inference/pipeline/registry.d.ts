/**
 * Pipeline Registry
 *
 * Maps modelType -> pipeline factory.
 *
 * @module inference/pipeline/registry
 */

import type { ManifestSchema } from '../../config/schema/index.js';

export type PipelineFactory = (manifest: ManifestSchema, contexts?: Record<string, unknown>) => Promise<unknown>;

export declare function registerPipeline(modelType: string, factory: PipelineFactory): void;
export declare function getPipelineFactory(modelType: string | null | undefined): PipelineFactory | null;
export declare function listPipelines(): string[];
