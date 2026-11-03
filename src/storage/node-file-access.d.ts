export interface NodeFileReader {
  readRange(offset?: number, length?: number | null): Promise<ArrayBuffer>;
  getSize(): Promise<number>;
  close(): Promise<void>;
}

export interface NodeFileAccess {
  readRange(fileRef: string | URL, offset?: number, length?: number | null): Promise<ArrayBuffer>;
  getSize(fileRef: string | URL): Promise<number>;
  close(): Promise<void>;
}

export declare function createNodeFileAccess(): NodeFileAccess;
