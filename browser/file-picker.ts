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

// ============================================================================
// Types
// ============================================================================

/**
 * File picker options for File System Access API
 */
interface FilePickerOptions {
  types?: Array<{
    description: string;
    accept: Record<string, string[]>;
  }>;
  multiple?: boolean;
}

/**
 * Directory picker options
 */
interface DirectoryPickerOptions {
  mode?: 'read' | 'readwrite';
  startIn?: 'desktop' | 'documents' | 'downloads' | 'music' | 'pictures' | 'videos';
}

/**
 * Result from picking files or a directory
 */
export interface PickResult {
  files: File[];
  directoryHandle?: FileSystemDirectoryHandle;
  directoryName?: string;
}

// ============================================================================
// File System Access API declarations
// ============================================================================

declare global {
  interface Window {
    showOpenFilePicker?: (options?: FilePickerOptions) => Promise<FileSystemFileHandle[]>;
    showDirectoryPicker?: (options?: DirectoryPickerOptions) => Promise<FileSystemDirectoryHandle>;
  }

  interface FileSystemDirectoryHandle {
    values(): AsyncIterableIterator<FileSystemHandle>;
  }

  interface HTMLInputElement {
    webkitdirectory: boolean;
  }
}

// ============================================================================
// Constants
// ============================================================================

const MODEL_FILE_EXTENSIONS = ['.gguf', '.safetensors', '.bin', '.json'];
const MODEL_FILE_TYPES = [
  {
    description: 'Model Files (GGUF, SafeTensors)',
    accept: {
      'application/octet-stream': ['.gguf', '.safetensors', '.bin'],
      'application/json': ['.json'],
    },
  },
];

// ============================================================================
// Public API
// ============================================================================

/**
 * Check if File System Access API is available
 */
export function hasFileSystemAccess(): boolean {
  return 'showOpenFilePicker' in window;
}

/**
 * Check if Directory Picker API is available
 */
export function hasDirectoryPicker(): boolean {
  return 'showDirectoryPicker' in window;
}

/**
 * Pick a single GGUF file (legacy API for backwards compatibility)
 * @returns The selected file, or null if cancelled
 */
export async function pickGGUFFile(): Promise<File | null> {
  const result = await pickModelFiles({ multiple: false });
  return result?.files[0] || null;
}

/**
 * Pick one or more model files (.gguf, .safetensors)
 * @param options.multiple - Allow selecting multiple files
 * @returns Array of selected files, or null if cancelled
 */
export async function pickModelFiles(options: { multiple?: boolean } = {}): Promise<PickResult | null> {
  const { multiple = true } = options;

  if (hasFileSystemAccess()) {
    return pickFilesWithFileSystemAccess(multiple);
  }
  return pickFilesWithFileInput(multiple);
}

/**
 * Pick a directory containing model files
 * @returns All model files in the directory, or null if cancelled
 */
export async function pickModelDirectory(): Promise<PickResult | null> {
  if (hasDirectoryPicker()) {
    return pickDirectoryWithFileSystemAccess();
  }
  // Fallback: use webkitdirectory attribute
  return pickDirectoryWithFileInput();
}

// ============================================================================
// File System Access API Implementation
// ============================================================================

/**
 * Pick files using File System Access API (Chrome/Edge)
 */
async function pickFilesWithFileSystemAccess(multiple: boolean): Promise<PickResult | null> {
  try {
    const fileHandles = await window.showOpenFilePicker!({
      types: MODEL_FILE_TYPES,
      multiple,
    });

    const files: File[] = [];
    for (const handle of fileHandles) {
      files.push(await handle.getFile());
    }

    return { files };
  } catch (err) {
    if ((err as Error).name === 'AbortError') {
      return null;
    }
    throw err;
  }
}

/**
 * Pick directory using File System Access API (Chrome/Edge)
 */
async function pickDirectoryWithFileSystemAccess(): Promise<PickResult | null> {
  try {
    const dirHandle = await window.showDirectoryPicker!({
      mode: 'read',
    });

    const files = await collectModelFilesFromDirectory(dirHandle);

    return {
      files,
      directoryHandle: dirHandle,
      directoryName: dirHandle.name,
    };
  } catch (err) {
    if ((err as Error).name === 'AbortError') {
      return null;
    }
    throw err;
  }
}

/**
 * Recursively collect model files from a directory handle
 */
async function collectModelFilesFromDirectory(
  dirHandle: FileSystemDirectoryHandle,
  maxDepth: number = 2
): Promise<File[]> {
  const files: File[] = [];

  for await (const entry of dirHandle.values()) {
    if (entry.kind === 'file') {
      const name = entry.name.toLowerCase();
      if (MODEL_FILE_EXTENSIONS.some(ext => name.endsWith(ext))) {
        const fileHandle = entry as FileSystemFileHandle;
        const file = await fileHandle.getFile();
        files.push(file);
      }
    } else if (entry.kind === 'directory' && maxDepth > 0) {
      // Recurse into subdirectories (but not too deep)
      const subDirHandle = entry as FileSystemDirectoryHandle;
      const subFiles = await collectModelFilesFromDirectory(subDirHandle, maxDepth - 1);
      files.push(...subFiles);
    }
  }

  return files;
}

// ============================================================================
// File Input Fallback Implementation
// ============================================================================

/**
 * Pick files using traditional file input (Firefox/Safari fallback)
 */
function pickFilesWithFileInput(multiple: boolean): Promise<PickResult | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = MODEL_FILE_EXTENSIONS.join(',');
    input.multiple = multiple;
    input.style.display = 'none';

    input.onchange = () => {
      const files = input.files ? Array.from(input.files) : [];
      cleanup();
      resolve(files.length > 0 ? { files } : null);
    };

    input.oncancel = () => {
      cleanup();
      resolve(null);
    };

    // Fallback for browsers without oncancel
    const handleFocusBack = (): void => {
      setTimeout(() => {
        if (document.body.contains(input) && !input.files?.length) {
          cleanup();
          resolve(null);
        }
      }, 300);
    };

    const cleanup = () => {
      window.removeEventListener('focus', handleFocusBack);
      if (document.body.contains(input)) {
        document.body.removeChild(input);
      }
    };

    window.addEventListener('focus', handleFocusBack, { once: true });
    document.body.appendChild(input);
    input.click();
  });
}

/**
 * Pick directory using webkitdirectory attribute (fallback)
 */
function pickDirectoryWithFileInput(): Promise<PickResult | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.webkitdirectory = true;
    input.style.display = 'none';

    input.onchange = () => {
      const allFiles = input.files ? Array.from(input.files) : [];
      // Filter to only model files
      const modelFiles = allFiles.filter(f =>
        MODEL_FILE_EXTENSIONS.some(ext => f.name.toLowerCase().endsWith(ext))
      );

      // Get directory name from path
      let directoryName: string | undefined;
      if (allFiles.length > 0 && allFiles[0].webkitRelativePath) {
        directoryName = allFiles[0].webkitRelativePath.split('/')[0];
      }

      cleanup();
      resolve(modelFiles.length > 0 ? { files: modelFiles, directoryName } : null);
    };

    input.oncancel = () => {
      cleanup();
      resolve(null);
    };

    const handleFocusBack = (): void => {
      setTimeout(() => {
        if (document.body.contains(input) && !input.files?.length) {
          cleanup();
          resolve(null);
        }
      }, 300);
    };

    const cleanup = () => {
      window.removeEventListener('focus', handleFocusBack);
      if (document.body.contains(input)) {
        document.body.removeChild(input);
      }
    };

    window.addEventListener('focus', handleFocusBack, { once: true });
    document.body.appendChild(input);
    input.click();
  });
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Check if streaming read is available (for large files)
 */
export function canStreamFile(file: File): boolean {
  return typeof file.stream === 'function';
}

/**
 * Get a readable stream from a file
 */
export function getFileStream(file: File): ReadableStream<Uint8Array> {
  if (!canStreamFile(file)) {
    throw new Error('File streaming not supported in this browser');
  }
  return file.stream();
}

/**
 * Group files by type (weights, config, tokenizer)
 */
export function categorizeModelFiles(files: File[]): {
  weights: File[];
  config: File | null;
  tokenizer: File | null;
  other: File[];
} {
  const result = {
    weights: [] as File[],
    config: null as File | null,
    tokenizer: null as File | null,
    other: [] as File[],
  };

  for (const file of files) {
    const name = file.name.toLowerCase();
    if (name.endsWith('.gguf') || name.endsWith('.safetensors') || name.endsWith('.bin')) {
      // Skip non-weight bins like tokenizer.bin
      if (name === 'tokenizer.bin' || name === 'tokenizer_config.bin') {
        result.other.push(file);
      } else {
        result.weights.push(file);
      }
    } else if (name === 'config.json' || name === 'model_config.json') {
      result.config = file;
    } else if (name === 'tokenizer.json' || name === 'tokenizer_config.json') {
      result.tokenizer = file;
    } else {
      result.other.push(file);
    }
  }

  // Sort weight files for proper shard ordering
  result.weights.sort((a, b) => a.name.localeCompare(b.name));

  return result;
}

/**
 * Detect model format from files
 */
export function detectModelFormat(files: File[]): 'gguf' | 'safetensors' | 'unknown' {
  for (const file of files) {
    const name = file.name.toLowerCase();
    if (name.endsWith('.gguf')) return 'gguf';
    if (name.endsWith('.safetensors')) return 'safetensors';
  }
  return 'unknown';
}

export default pickGGUFFile;
