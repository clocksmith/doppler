export * from '../config/pack-v2.js';
import type { DopplerPackV2 } from '../config/pack-v2.js';

export declare function writePackV2(outputPath: string, pack: DopplerPackV2): Promise<{
  ok: true;
  outputPath: string;
  semanticRoot: string;
  envelopeHash: string;
}>;
export declare function loadPackV2(packPath: string, options?: { requireSignature?: boolean }): Promise<DopplerPackV2>;
export declare function loadPackSigningKey(value: string): Promise<JsonWebKey>;
