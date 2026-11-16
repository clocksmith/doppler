export { destroyPleRuntimeCache, createPleBufferCache, destroyPleBufferCache, getPleHotVocabularyRuntime, prefetchPerLayerRow, hasRangeBackedPerLayerInputEmbeddings, hasGpuSplitPerLayerInputEmbeddings } from "./per-layer-inputs.js";
export function getEmbeddingSource(weight: unknown, label: string): GPUBuffer | Float32Array | import('../../../../gpu/weight-buffer.js').CpuWeightBuffer;
export function getEmbeddingDtype(weight: unknown): string | null;
export function getEmbeddingTranspose(weight: unknown): boolean;
export function releasePleSliceBuffer(
  recorder: import('../../../../gpu/command-recorder.js').CommandRecorder | null | undefined,
  buffer: GPUBuffer | null | undefined,
  decodeBuffers: import('../../../decode-buffers.js').DecodeBufferManager | null | undefined,
  cache: import("./per-layer-inputs.js").PleBufferCache | null | undefined
): void;
