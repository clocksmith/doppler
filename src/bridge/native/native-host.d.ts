#!/usr/bin/env node
/**
 * DOPPLER Native Bridge Host - Native messaging protocol for Chrome extension.
 * Provides file access to bypass browser storage limits.
 */

export interface BinaryMessage {
  type: 'binary';
  data: number[];
}

export interface AckMessage {
  type: 'ack';
}

export type NativeMessage = BinaryMessage | AckMessage;

export interface ListEntry {
  name: string;
  isDir: boolean;
  size: number;
}

export declare function isPathAllowed(filePath: string): boolean;

export declare function handleMessage(msg: NativeMessage): Promise<BinaryMessage[]>;

export declare function handleReadRequest(reqId: number, payload: Buffer): Promise<BinaryMessage[]>;

export declare function handleListRequest(reqId: number, payload: Buffer): Promise<BinaryMessage[]>;
