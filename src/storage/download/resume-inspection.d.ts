import type { RDRRManifest } from '../../formats/rdrr/types.js';

export declare const MODEL_DOWNLOAD_RESUME_INSPECTION_SCHEMA: 'doppler.model-download-resume-inspection.v1';
export { buildManifestVersionSet } from './integrity.js';

export interface ModelDownloadResumeInspection {
  schemaVersion: typeof MODEL_DOWNLOAD_RESUME_INSPECTION_SCHEMA;
  modelId: string;
  manifestVersionSet: string;
  statePresent: boolean;
  manifestMatched: boolean;
  totalBytes: number;
  verifiedBytes: number;
  remainingBytes: number;
  totalShards: number;
  verifiedShards: number;
}

export declare function inspectModelDownloadResume(
  modelId: string,
  manifest: RDRRManifest
): Promise<ModelDownloadResumeInspection>;
