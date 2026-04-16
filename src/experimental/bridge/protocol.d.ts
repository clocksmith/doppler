/**
 * DOPPLER Native Bridge Protocol
 * Binary protocol for Extension <-> Native Host communication.
 */

export declare const MAGIC: number;
export declare const HEADER_SIZE: number;
export declare const MAX_CHUNK_SIZE: number;
export declare const ACK_SIZE: number;

export declare const Command: {
  readonly PING: 0x00;
  readonly PONG: 0x01;
  readonly READ: 0x02;
  readonly READ_RESPONSE: 0x03;
  readonly WRITE: 0x04;
  readonly WRITE_ACK: 0x05;
  readonly LIST: 0x06;
  readonly LIST_RESPONSE: 0x07;
  readonly ERROR: 0xff;
};

export type CommandType = (typeof Command)[keyof typeof Command];

export declare const Flag: {
  readonly NONE: 0x00;
  readonly COMPRESSED: 0x01;
  readonly LAST_CHUNK: 0x02;
};

export type FlagType = (typeof Flag)[keyof typeof Flag];

export declare const ErrorCode: {
  readonly OK: 0;
  readonly NOT_FOUND: 1;
  readonly PERMISSION_DENIED: 2;
  readonly IO_ERROR: 3;
  readonly INVALID_REQUEST: 4;
  readonly QUOTA_EXCEEDED: 5;
};

export type ErrorCodeType = (typeof ErrorCode)[keyof typeof ErrorCode];

export interface MessageHeader {
  cmd: CommandType;
  flags: FlagType;
  reqId: number;
  payloadLen: number;
}

export interface ReadResponse {
  offset: number;
  data: Uint8Array;
}

export interface ErrorResponse {
  code: ErrorCodeType;
  message: string;
}

export interface ListEntry {
  name: string;
  isDir: boolean;
  size: number;
}

export declare function encodeMessage(
  cmd: CommandType,
  reqId: number,
  payload?: Uint8Array | null,
  flags?: FlagType
): ArrayBuffer;

export declare function decodeHeader(buffer: ArrayBuffer): MessageHeader | null;

export declare function createReadRequest(
  reqId: number,
  path: string,
  offset: number,
  length: number
): ArrayBuffer;

export declare function createListRequest(reqId: number, path: string): ArrayBuffer;

export declare function createAck(reqId: number): ArrayBuffer;

export declare function parseReadResponse(payload: Uint8Array): ReadResponse;

export declare function parseListResponse(payload: Uint8Array): ListEntry[];

export declare function parseErrorResponse(payload: Uint8Array): ErrorResponse;

// Legacy aliases
export declare const CMD: typeof Command;
export declare const FLAGS: typeof Flag;
export declare const ERROR_CODES: typeof ErrorCode;
