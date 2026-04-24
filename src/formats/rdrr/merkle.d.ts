export interface MerkleProofStep {
  position: 'left' | 'right';
  hash: string;
}

export interface MerkleTree {
  blockSize: number;
  blockCount: number;
  leafHashes: string[];
  levels: string[][];
  root: string;
}

export declare const DEFAULT_MERKLE_BLOCK_SIZE: number;

export declare function splitIntoMerkleBlocks(bytes: Uint8Array | ArrayBuffer, blockSize: number): Uint8Array[];
export declare function buildMerkleTree(bytes: Uint8Array | ArrayBuffer, options?: { blockSize?: number }): MerkleTree;
export declare function buildMerkleTreeFromLeafHashes(leafHashes: string[], options?: { blockSize?: number }): MerkleTree;
export declare function buildMerkleProof(tree: MerkleTree, blockIndex: number): MerkleProofStep[];
export declare function verifyMerkleProof(options: {
  blockBytes: Uint8Array | ArrayBuffer;
  proof: MerkleProofStep[];
  expectedRoot: string;
}): boolean;
