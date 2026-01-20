export interface Blake3Hasher {
  update(data: Uint8Array | ArrayBuffer): void;
  finalize(): Uint8Array;
}

export declare function createHasher(): Blake3Hasher;

export declare function hash(data: Uint8Array | ArrayBuffer): Promise<Uint8Array>;
