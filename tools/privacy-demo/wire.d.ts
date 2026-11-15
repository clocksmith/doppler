import type { Readable, Writable } from 'node:stream';
export declare function fail(): never;
export declare function envelope(session: string, keysId: string, step: number, integrityMode?: 'none' | 'client-reference' | 'ciphertext-replay'): Record<string, unknown>;
export declare function validate(value: unknown, binding: Record<string, unknown>, field: string): string;
export declare function readFrames(stream: Readable | AsyncIterable<Uint8Array>): AsyncGenerator<Record<string, unknown>>;
export declare function writeFrame(stream: Writable, value: unknown): Promise<number>;
