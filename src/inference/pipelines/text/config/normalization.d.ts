import type { AudioFeatureExtractorConfig } from '../config.js';

export declare const resolveAudioConfig: any;
export declare const resolvePerLayerInputsSession: any;
export declare const resolveSessionSettings: any;
export declare const resolveVisionConfig: any;

export declare function resolveAudioFeatureExtractor(raw: unknown, modelId?: string): AudioFeatureExtractorConfig | null;
