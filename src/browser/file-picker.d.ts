/**
 * file-picker.ts - Model File/Folder Picker
 *
 * Supports:
 * - Single file selection (.gguf, .safetensors)
 * - Multiple file selection (for sharded models)
 * - Directory selection (pick a model folder)
 *
 * Uses File System Access API on Chrome/Edge, falls back to <input type="file">
 * for Firefox/Safari.
 *
 * @module browser/file-picker
 */

/**
 * Result from picking files or a directory
 */
export interface PickResult {
  files: File[];
  directoryHandle?: FileSystemDirectoryHandle;
  directoryName?: string;
}

/**
 * Check if File System Access API is available
 */
export declare function hasFileSystemAccess(): boolean;

/**
 * Check if Directory Picker API is available
 */
export declare function hasDirectoryPicker(): boolean;

/**
 * Pick a single GGUF file (legacy API for backwards compatibility)
 * @returns The selected file, or null if cancelled
 */
export declare function pickGGUFFile(): Promise<File | null>;

/**
 * Pick one or more model files (.gguf, .safetensors)
 * @param options.multiple - Allow selecting multiple files
 * @returns Array of selected files, or null if cancelled
 */
export declare function pickModelFiles(options?: { multiple?: boolean }): Promise<PickResult | null>;

/**
 * Pick a directory containing model files
 * @returns All model files in the directory, or null if cancelled
 */
export declare function pickModelDirectory(): Promise<PickResult | null>;

/**
 * Check if streaming read is available (for large files)
 */
export declare function canStreamFile(file: File): boolean;

/**
 * Detect model format from files
 */
export declare function detectModelFormat(files: File[]): 'gguf' | 'safetensors' | 'unknown';

export default pickGGUFFile;
