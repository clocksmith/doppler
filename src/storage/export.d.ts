export interface ExportProgress {
  stage: string;
  modelId: string;
  backend?: string;
  filename?: string;
  index?: number;
  total?: number;
  fileCount?: number;
  writtenBytes?: number;
}

/** Exports from the configured persistent backend; rejects memory storage and invalid chunk sizes. */
export declare function exportModelToDirectory(
  modelId: string,
  destinationDir: FileSystemDirectoryHandle,
  options?: {
    onProgress?: (progress: ExportProgress) => void;
    chunkBytes?: number;
  }
): Promise<{ modelId: string; fileCount: number }>;

