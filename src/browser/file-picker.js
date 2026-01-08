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
export function hasFileSystemAccess() {
  return 'showOpenFilePicker' in window;
}

/**
 * Check if Directory Picker API is available
 */
export function hasDirectoryPicker() {
  return 'showDirectoryPicker' in window;
}

/**
 * Pick a single GGUF file (legacy API for backwards compatibility)
 * @returns The selected file, or null if cancelled
 */
export async function pickGGUFFile() {
  const result = await pickModelFiles({ multiple: false });
  return result?.files[0] || null;
}

/**
 * Pick one or more model files (.gguf, .safetensors)
 * @param options.multiple - Allow selecting multiple files
 * @returns Array of selected files, or null if cancelled
 */
export async function pickModelFiles(options = {}) {
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
export async function pickModelDirectory() {
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
async function pickFilesWithFileSystemAccess(multiple) {
  try {
    const fileHandles = await window.showOpenFilePicker({
      types: MODEL_FILE_TYPES,
      multiple,
    });

    const files = [];
    for (const handle of fileHandles) {
      files.push(await handle.getFile());
    }

    return { files };
  } catch (err) {
    if (err.name === 'AbortError') {
      return null;
    }
    throw err;
  }
}

/**
 * Pick directory using File System Access API (Chrome/Edge)
 */
async function pickDirectoryWithFileSystemAccess() {
  try {
    const dirHandle = await window.showDirectoryPicker({
      mode: 'read',
    });

    const files = await collectModelFilesFromDirectory(dirHandle);

    return {
      files,
      directoryHandle: dirHandle,
      directoryName: dirHandle.name,
    };
  } catch (err) {
    if (err.name === 'AbortError') {
      return null;
    }
    throw err;
  }
}

/**
 * Recursively collect model files from a directory handle
 */
async function collectModelFilesFromDirectory(
  dirHandle,
  maxDepth = 2
) {
  const files = [];

  for await (const entry of dirHandle.values()) {
    if (entry.kind === 'file') {
      const name = entry.name.toLowerCase();
      if (MODEL_FILE_EXTENSIONS.some(ext => name.endsWith(ext))) {
        const fileHandle = entry;
        const file = await fileHandle.getFile();
        files.push(file);
      }
    } else if (entry.kind === 'directory' && maxDepth > 0) {
      // Recurse into subdirectories (but not too deep)
      const subDirHandle = entry;
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
function pickFilesWithFileInput(multiple) {
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
    const handleFocusBack = () => {
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
function pickDirectoryWithFileInput() {
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
      let directoryName;
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

    const handleFocusBack = () => {
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
export function canStreamFile(file) {
  return typeof file.stream === 'function';
}

/**
 * Detect model format from files
 */
export function detectModelFormat(files) {
  for (const file of files) {
    const name = file.name.toLowerCase();
    if (name.endsWith('.gguf')) return 'gguf';
    if (name.endsWith('.safetensors')) return 'safetensors';
  }
  return 'unknown';
}

export default pickGGUFFile;
