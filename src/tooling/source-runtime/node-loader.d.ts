import type { Stats } from 'node:fs';
import type { ParsedSafetensorsHeader } from '../../formats/safetensors/types.js';

export const MAX_NODE_READ_BYTES: number;

export interface NodeFileAccess {
  readRange(filePath: string, offset: number, length: number): Promise<ArrayBuffer>;
  getSize(filePath: string): Promise<number>;
  close(): Promise<void>;
}

export interface NodeFileReaders {
  readRange(filePath: string, offset: number, length: number): Promise<ArrayBuffer>;
  streamRange(
    filePath: string,
    offset: number,
    length: number,
    options?: { chunkBytes?: number }
  ): AsyncGenerator<Uint8Array>;
  readText(filePath: string): Promise<string | null>;
  readBinary(filePath: string): Promise<ArrayBuffer>;
  close(): Promise<void>;
}

export interface SourceFileEntry {
  path: string;
  size?: number;
  hash?: string;
  hashAlgorithm?: string;
  [key: string]: unknown;
}

export function normalizePath(value: unknown): string;
export function isGgufPath(filePath: unknown): boolean;
export function isTflitePath(filePath: unknown): boolean;
export function isLiteRTTaskPath(filePath: unknown): boolean;
export function isLiteRTLMPath(filePath: unknown): boolean;
export function getPathStats(targetPath: string, label: string): Promise<Stats>;
export function fileExists(targetPath: string): Promise<boolean>;
export function readJson(filePath: string, label: string): Promise<Record<string, unknown>>;
export function createNodeFileAccess(): NodeFileAccess;
export function readSafetensorsHeaderFromFile(
  filePath: string,
  fileAccess: NodeFileAccess
): Promise<ParsedSafetensorsHeader>;
export function buildNodeFileReaders(fileAccess: NodeFileAccess): NodeFileReaders;
export function addHashesToFileEntries(
  entries: SourceFileEntry[],
  hashAlgorithm: string
): Promise<SourceFileEntry[]>;
