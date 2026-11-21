import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { lstat, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { createVerifiedPieceStorage } from '../../src/storage/verified-piece-storage.js';

const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));

test('idle background piece verification prioritizes in-stage routed expert spans ahead of non-expert prefix pieces', async () => {
  const root = await mkdtemp(join(tmpdir(), 'expert-priority-bg-'));
  try {
    const count = 16;
    const pieceSize = 64;
    const pieces = [];
    const data = new Uint8Array(count * pieceSize);
    for (let i = 0; i < count; i++) {
      const slice = data.subarray(i * pieceSize, (i + 1) * pieceSize);
      for (let j = 0; j < pieceSize; j++) slice[j] = (i * 37 + j * 13 + 7) & 0xff;
      pieces.push({ offset: i * pieceSize, size: pieceSize, identity: sha(slice) });
    }
    // Out-of-stage expert at piece 2 (layer 0), non-expert dense weights across pieces 0..11,
    // and in-stage routed experts at pieces 12..13 (layer 2).
    const manifestBytes = encode({
      hashAlgorithm: 'sha256',
      shards: [{ filename: 'weights.bin', path: 'weights.bin', size: data.byteLength, hash: sha(data).slice(7), hashAlgorithm: 'sha256' }],
      tensors: {
        'model.embed_tokens.weight': { shard: 0, offset: 0, size: 128 },
        'blk.0.ffn_gate_exps.weight': { shard: 0, offset: 2 * pieceSize, size: pieceSize },
        'model.layers.1.self_attn.q_proj.weight': { shard: 0, offset: 4 * pieceSize, size: 4 * pieceSize },
        'blk.2.ffn_gate_exps.weight': { shard: 0, offset: 12 * pieceSize, size: 2 * pieceSize },
      },
    });
    const indexBytes = encode({
      schema: 'doppler.verified-pieces/v1',
      manifestIdentity: sha(manifestBytes),
      maxPieceBytes: pieceSize,
      files: [{ path: 'weights.bin', size: data.byteLength, pieces }],
    });
    const weightPath = join(root, 'weights.bin');
    await writeFile(weightPath, data);
    const st = await lstat(weightPath, { bigint: true });

    const opened = await createVerifiedPieceStorage({
      manifestBytes,
      indexBytes,
      indexIdentity: sha(indexBytes),
      acquire: async piece => (await readFile(weightPath)).subarray(piece.offset, piece.offset + piece.size),
      nodeGather: {
        files: { 'weights.bin': weightPath },
        fileIdentities: {
          'weights.bin': { device: st.dev.toString(), inode: st.ino.toString(), size: st.size.toString() },
        },
        maxPartBytes: 128,
        maxRanges: 3,
        memoryBudgetBytes: 128 + 128,
        concurrency: 1,
        verificationCache: {
          backgroundVerify: true,
          backgroundIdleDelayMs: 15,
          priorityLayerRange: [2, 3],
        },
      },
    });
    try {
      const deadline = Date.now() + 2000;
      while (!opened.storage.rangeLease.hasVerified(0, 12 * pieceSize, 2 * pieceSize)) {
        if (Date.now() > deadline) throw new Error('Timed out waiting for priority routed expert verification.');
        await new Promise(resolve => setTimeout(resolve, 5));
      }
      // At the moment pieces 12..13 finish verifying, later non-priority pieces (e.g. piece 8) must still be unverified,
      // proving routed experts inside [2, 3) jumped to the front of bgQueue ahead of linear file order and layer 0.
      assert.equal(opened.storage.rangeLease.hasVerified(0, 12 * pieceSize, 2 * pieceSize), true);
      assert.equal(opened.storage.rangeLease.hasVerified(0, 8 * pieceSize, pieceSize), false);
    } finally {
      await opened.storage.close();
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
