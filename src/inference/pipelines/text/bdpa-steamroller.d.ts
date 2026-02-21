/**
 * BDPA steamroller host-side preprocessing helpers.
 */

export interface BDPAGeneratedData {
  tBasisK: Float32Array;
  tBasisV: Float32Array;
  pDeltaK: Int8Array;
  pDeltaV: Int8Array;
  iFlat: Uint32Array;
  numBasisVectors: number;
}

export declare function steamrollerRadixArgsort(tokens: Int32Array | Uint32Array): Uint32Array;

export declare function generateBDPAData(
  sortedIndices: Uint32Array,
  kBuffer: Float32Array | Uint16Array,
  vBuffer: Float32Array | Uint16Array,
  tokens: Int32Array | Uint32Array,
  kvSize: number
): BDPAGeneratedData;
