import type { HashAlgorithm, RDRRManifest } from '../../formats/rdrr/types.js';

export interface StreamingHasher {
  update(data: Uint8Array | ArrayBuffer): void;
  finalize(): Promise<Uint8Array>;
}

export interface TensorIntegrityController {
  reset(): void;
  verifyTensorRoot(
    manifest: RDRRManifest,
    tensorId: string
  ): Promise<{ tensorId: string; location: Record<string, unknown>; expectedRoot: string }>;
  verifyTensorRange(
    manifest: RDRRManifest,
    shardIndex: number,
    offset: number,
    length: number | null,
    tensorId: string
  ): Promise<void>;
}

export function getHashAlgorithm(): HashAlgorithm | null;
export function hexToBytes(hex: string): Uint8Array;
export function computeBlake3(data: Uint8Array | ArrayBuffer): Promise<string>;
export function computeSHA256(data: Uint8Array | ArrayBuffer): Promise<string>;
export function computeHash(data: Uint8Array | ArrayBuffer, algorithm: HashAlgorithm): Promise<string>;
export function createStreamingHasher(algorithm: HashAlgorithm): Promise<StreamingHasher>;
export function requireManifestHashAlgorithm(manifest: RDRRManifest, context: string): HashAlgorithm;
export function createTensorIntegrityController(dependencies: {
  readBackendFileRange(filename: string, offset: number, length: number | null): Promise<ArrayBuffer>;
  loadTensorsFromStore(): Promise<string | null>;
}): TensorIntegrityController;
