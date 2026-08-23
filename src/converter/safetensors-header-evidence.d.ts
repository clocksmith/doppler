export const SAFETENSORS_HEADER_PIN_SCHEMA_ID: 'doppler.safetensors-header-pin/v1';
export const SAFETENSORS_HEADER_EVIDENCE_SCHEMA_ID: 'doppler.safetensors-header-evidence/v1';

export interface SafetensorsHeaderEvidence {
  sourceFile: string;
  sourceHeaderSha256: string;
  headerLength: number;
  tensorCount: number;
  tensors: Record<string, { dtype: string; shape: number[]; sourceFile: string }>;
}

export declare function readSafetensorsHeaderLength(prefix: Uint8Array): number;

export declare function parseSafetensorsHeaderEvidence(
  bytes: Uint8Array,
  options: { sourceFile: string; expectedSha256: string }
): SafetensorsHeaderEvidence;

export declare function materializeSafetensorsHeaderEvidence(
  pin: Record<string, unknown>,
  readRange: (request: {
    repository: string;
    revision: string;
    sourceFile: string;
    start: number;
    end: number;
  }) => Promise<Uint8Array>
): Promise<Record<string, unknown>>;

export declare function validateSafetensorsIndexEvidence(
  headers: Record<string, unknown>,
  index: Record<string, unknown>
): Record<string, unknown>;
