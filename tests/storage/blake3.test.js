import assert from 'node:assert/strict';

const { createHasher, hash } = await import('../../src/storage/blake3.js');

const bytes = new Uint8Array([1, 2, 3, 4]);
const hasher = createHasher();
hasher.update(bytes.subarray(0, 2));
hasher.update(bytes.subarray(2));

const first = hasher.finalize();
const second = hasher.finalize();
const expected = await hash(bytes);

assert.deepEqual(first, expected);
assert.deepEqual(second, expected);
assert.notEqual(first, second, 'finalize should return a defensive copy');
assert.throws(
  () => hasher.update(new Uint8Array([5])),
  /update called after finalize/
);

const emptyHasher = createHasher();
assert.deepEqual(emptyHasher.finalize(), await hash(new Uint8Array(0)));
assert.deepEqual(emptyHasher.finalize(), await hash(new Uint8Array(0)));

// Retained digest values from 7ac72347. These protect existing artifact identity;
// they are not an assertion of compatibility with another hash implementation.
const retained = [
  {
    "length": 0,
    "hex": "fda1200e7626ed65703a9c8eb28987d8142bf29fcf7cf582e0c44cc7ad8a8ea8"
  },
  {
    "length": 1,
    "hex": "1103c1e16280c2008da4b71450ad5ad844b71a5e09df9c1948dfb83f4a77e0d9"
  },
  {
    "length": 63,
    "hex": "04945a4d1882b21682e02a401f75b06487ae3cce3efd7fdf6543bf4d5e47c8a3"
  },
  {
    "length": 64,
    "hex": "69d0b52250aa4e600687b20755fea4ba76a9fc25490d015d33ded4c09f9e348a"
  },
  {
    "length": 65,
    "hex": "b5ef4905c6aa739c7c0573dc76813e0b6ef2da83aafbc212bb4e30181783671f"
  },
  {
    "length": 1023,
    "hex": "5a8ef1b30754857dd8b96ef4b49a808271a0c1bc31fb502c0c99c0973f89ec31"
  },
  {
    "length": 1024,
    "hex": "720a04b59cf6a8b148ca1f6ef8a3b2168077f2baf111197742bc6b17e2dbc694"
  },
  {
    "length": 1025,
    "hex": "b74c0de1b15fd2449d0ca9d16e22a356275308cc3ca15c34659c6773168a5179"
  },
  {
    "length": 2048,
    "hex": "64ed429b3b87910734e0320f42b34e8fd854a41920f4cc9ea1e24a2d35d92293"
  },
  {
    "length": 3073,
    "hex": "e503b45df53e8819cb6c7a3052889aceb321a5f52af45194fbaccee83dd53213"
  },
  {
    "length": 65536,
    "hex": "ea7c8530499c2794adde9c51055956338e7bef1a4b771cc71ebb27a026d323f0"
  },
  {
    "length": 1048577,
    "hex": "0d831c3cc126c01dd7abfec563f750b1c8d8dd1358b734645564c513e785d42f"
  }
];
for (const row of retained) {
  const input = Uint8Array.from({ length: row.length }, (_, index) => index % 251);
  assert.equal(Buffer.from(await hash(input)).toString('hex'), row.hex);
  for (const step of [1, 777, 65536]) {
    const incremental = createHasher();
    for (let offset = 0; offset < input.length; offset += step) incremental.update(input.subarray(offset, offset + step));
    assert.equal(Buffer.from(incremental.finalize()).toString('hex'), row.hex);
  }
}
console.log('blake3.test: ok');
