import { hashBytesSha256 } from '../../utils/canonical-hash.js';

export const DEFAULT_MERKLE_BLOCK_SIZE = 1024 * 1024;

function digestHexToBytes(digest) {
  const normalized = String(digest || '').trim();
  if (!normalized.startsWith('sha256:')) {
    throw new Error(`[RDRRMerkle] unsupported digest "${digest}"`);
  }
  const hex = normalized.slice(7);
  if (hex.length % 2 !== 0 || !/^[0-9a-fA-F]*$/.test(hex)) {
    throw new Error(`[RDRRMerkle] invalid hex in digest "${digest}"`);
  }
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i += 1) {
    bytes[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
  }
  return bytes;
}

function combineHashes(left, right) {
  const combined = new Uint8Array(64);
  combined.set(digestHexToBytes(left), 0);
  combined.set(digestHexToBytes(right), 32);
  return hashBytesSha256(combined);
}

export function splitIntoMerkleBlocks(bytes, blockSize) {
  if (!Number.isInteger(blockSize) || blockSize <= 0) {
    throw new Error('[RDRRMerkle] blockSize must be a positive integer');
  }
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (view.byteLength === 0) {
    return [new Uint8Array(0)];
  }
  const blocks = [];
  for (let offset = 0; offset < view.byteLength; offset += blockSize) {
    blocks.push(view.slice(offset, Math.min(view.byteLength, offset + blockSize)));
  }
  return blocks;
}

export function buildMerkleTree(bytes, options = {}) {
  const blockSize = options.blockSize ?? DEFAULT_MERKLE_BLOCK_SIZE;
  const blocks = splitIntoMerkleBlocks(bytes, blockSize);
  const leafHashes = blocks.map((block) => hashBytesSha256(block));
  return buildMerkleTreeFromLeafHashes(leafHashes, { blockSize });
}

export function buildMerkleTreeFromLeafHashes(leafHashesInput, options = {}) {
  const blockSize = options.blockSize ?? DEFAULT_MERKLE_BLOCK_SIZE;
  const leafHashes = Array.isArray(leafHashesInput) ? [...leafHashesInput] : [];
  if (leafHashes.length === 0) {
    leafHashes.push(hashBytesSha256(new Uint8Array(0)));
  }
  const levels = [leafHashes];
  while (levels[levels.length - 1].length > 1) {
    const current = levels[levels.length - 1];
    const next = [];
    for (let index = 0; index < current.length; index += 2) {
      const left = current[index];
      const right = current[index + 1] ?? current[index];
      next.push(combineHashes(left, right));
    }
    levels.push(next);
  }
  return {
    blockSize,
    blockCount: leafHashes.length,
    leafHashes,
    levels,
    root: levels[levels.length - 1][0],
  };
}

export function buildMerkleProof(tree, blockIndex) {
  if (!tree || !Array.isArray(tree.levels)) {
    throw new Error('[RDRRMerkle] buildMerkleProof requires a merkle tree');
  }
  if (!Number.isInteger(blockIndex) || blockIndex < 0 || blockIndex >= tree.leafHashes.length) {
    throw new Error(`[RDRRMerkle] invalid block index ${blockIndex}`);
  }
  const proof = [];
  let cursor = blockIndex;
  for (let levelIndex = 0; levelIndex < tree.levels.length - 1; levelIndex += 1) {
    const level = tree.levels[levelIndex];
    const siblingIndex = cursor % 2 === 0 ? cursor + 1 : cursor - 1;
    const siblingHash = level[siblingIndex] ?? level[cursor];
    proof.push({
      position: cursor % 2 === 0 ? 'right' : 'left',
      hash: siblingHash,
    });
    cursor = Math.floor(cursor / 2);
  }
  return proof;
}

export function verifyMerkleProof(options) {
  const blockBytes = options?.blockBytes;
  const proof = Array.isArray(options?.proof) ? options.proof : [];
  const expectedRoot = String(options?.expectedRoot || '').trim();
  let current = hashBytesSha256(blockBytes instanceof Uint8Array ? blockBytes : new Uint8Array(blockBytes));
  for (const step of proof) {
    if (!step || typeof step !== 'object') {
      throw new Error('[RDRRMerkle] proof step must be an object');
    }
    if (step.position === 'left') {
      current = combineHashes(step.hash, current);
      continue;
    }
    if (step.position === 'right') {
      current = combineHashes(current, step.hash);
      continue;
    }
    throw new Error(`[RDRRMerkle] invalid proof position "${step.position}"`);
  }
  return current === expectedRoot;
}
