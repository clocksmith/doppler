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
