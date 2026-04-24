import assert from 'node:assert/strict';

import {
  buildMerkleProof,
  buildMerkleTree,
  verifyMerkleProof,
} from '../../src/formats/rdrr/merkle.js';

{
  const bytes = new TextEncoder().encode('doppler-rdrr-merkle-test');
  const tree = buildMerkleTree(bytes, { blockSize: 8 });
  assert.equal(tree.blockCount, 3);
  assert.ok(tree.root.startsWith('sha256:'));
  const proof = buildMerkleProof(tree, 1);
  const block = bytes.slice(8, 16);
  assert.equal(verifyMerkleProof({
    blockBytes: block,
    proof,
    expectedRoot: tree.root,
  }), true);
}

{
  const bytes = new TextEncoder().encode('single-block');
  const tree = buildMerkleTree(bytes, { blockSize: 4096 });
  assert.equal(tree.blockCount, 1);
  assert.equal(tree.root, tree.leafHashes[0]);
}

console.log('rdrr-merkle.test: ok');

