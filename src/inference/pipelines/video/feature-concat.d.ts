/**
 * Plan the copies that concatenate per-frame vision features into one buffer.
 * Every frame must declare the same positive featureSize and a positive token count.
 */
export function planVideoFeatureConcat(frameResults: Array<{
    featureSize: number;
    numTokens: number;
}>): {
    featureSize: number;
    totalTokens: number;
    totalBytes: number;
    copies: Array<{ offset: number; byteLength: number }>;
};
