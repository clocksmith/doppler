export declare const calculateAttentionWorkgroups: any;
export declare const getChunkedMaxKVLen: any;
export declare const getRequiredVariantMaxKVLen: any;
export declare const inferAttentionTierFromVariant: any;
export declare const loggedAttentionTier: any;
export declare const loggedChunkedKernel: any;
export declare const resolveAttentionPlan: any;
export declare const resolveAttentionQueryBlockSize: any;
export declare const resolveAttentionVariant: any;
export declare const selectAttentionTier: any;
export declare const validateAttentionVariant: any;

export declare function resolveOnlineDecodePipelineConstants(
  variant: string | null | undefined,
  headDim: number,
  policy: { workgroupSize?: number | null; useDirectContiguousKVLayout?: boolean } | null | undefined,
  options?: { kvLayout?: string; slidingWindow?: number },
): { WORKGROUP_SIZE?: number; USE_DIRECT_KV_LAYOUT?: boolean } | null;

/**
 * Declare to a step observer of `recorder` the kvLen bound within which this decode
 * step's attention selection holds: the chunked cap, the variant's maxKVLen and an
 * unfilled sliding window (kvStart 0).
 */
export declare function declareDecodeAttentionLimit(
  recorder: import('../../command-recorder.js').CommandRecorder | null,
  variant: string,
  kvLen: number,
  kvStart: number,
  slidingWindow: number | null
): void;
