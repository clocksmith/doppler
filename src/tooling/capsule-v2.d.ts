export * from '../config/capsule-v2.js';
import type { DopplerCapsuleV2 } from '../config/capsule-v2.js';

export declare function writeCapsuleV2(outputPath: string, capsule: DopplerCapsuleV2): Promise<{
  ok: true;
  outputPath: string;
  semanticRoot: string;
  envelopeHash: string;
}>;
export declare function loadCapsuleV2(capsulePath: string, options?: { requireSignature?: boolean }): Promise<DopplerCapsuleV2>;
export declare function loadCapsuleSigningKey(value: string): Promise<JsonWebKey>;
