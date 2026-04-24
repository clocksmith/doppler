import type { IntegrityExtensions, RDRRManifest, TensorMap, TensorLocation } from './types.js';

export declare const DEFAULT_MERKLE_BLOCK_SIZE: number;

export interface TensorBlockMerkleRoot {
  tensorId: string;
  blockSize: number;
  blockCount: number;
  totalBytes: number;
  root: string;
}

export interface IntegrityBuildProgress {
  tensorId: string;
  current: number;
  total: number;
}

export declare function buildTensorBlockMerkleRoot(
  tensorId: string,
  location: TensorLocation,
  options: {
    blockSize?: number;
    readShardRange: (
      shardIndex: number,
      offset: number,
      length: number,
      context?: { tensorId: string }
    ) => Promise<Uint8Array | ArrayBuffer>;
    hashBlockBytesSha256?: (bytes: Uint8Array) => string;
  }
): Promise<TensorBlockMerkleRoot>;

export declare function buildIntegrityExtensions(
  manifest: RDRRManifest,
  options: {
    tensorMap?: TensorMap;
    blockSize?: number;
    onProgress?: (progress: IntegrityBuildProgress) => void;
    readShardRange: (
      shardIndex: number,
      offset: number,
      length: number,
      context?: { tensorId: string }
    ) => Promise<Uint8Array | ArrayBuffer>;
    hashBlockBytesSha256?: (bytes: Uint8Array) => string;
  }
): Promise<{
  integrityExtensions: IntegrityExtensions;
  integrityExtensionsHash: string;
}>;
