

// ============================================================================
// Constants
// ============================================================================

const MODEL_FILE_EXTENSIONS = ['.gguf', '.safetensors', '.bin', '.json', '.txt', '.model'];
const MODEL_DIRECTORY_MAX_DEPTH = 4;
const MODEL_FILE_TYPES = [
  {
    description: 'Model Files (GGUF, SafeTensors)',
    accept: {
      'application/octet-stream': ['.gguf', '.safetensors', '.bin', '.model'],
      'application/json': ['.json'],
      'text/plain': ['.txt'],
    },
  },
];

function attachRelativePath(file, relativePath) {
  if (!file || !relativePath) return;
  try {
    Object.defineProperty(file, 'relativePath', {
      value: relativePath,
      configurable: true,
    });
  } catch {
    // Ignore if File is non-extensible in this environment
  }
}

// ============================================================================
// Public API
// ============================================================================


export function hasFileSystemAccess() {
  return 'showOpenFilePicker' in window;
}


export function hasDirectoryPicker() {
  return 'showDirectoryPicker' in window;
}


export async function pickGGUFFile() {
  const result = await pickModelFiles({ multiple: false });
  return result?.files[0] || null;
}


export async function pickModelFiles(options = {}) {
  const { multiple = true } = options;

  if (hasFileSystemAccess()) {
    return pickFilesWithFileSystemAccess(multiple);
  }
  return pickFilesWithFileInput(multiple);
}


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


async function collectModelFilesFromDirectory(
  dirHandle,
  basePath = '',
  maxDepth = MODEL_DIRECTORY_MAX_DEPTH
) {
  const files = [];

  for await (const entry of dirHandle.values()) {
    if (entry.kind === 'file') {
      const name = entry.name.toLowerCase();
      if (MODEL_FILE_EXTENSIONS.some(ext => name.endsWith(ext))) {
        const fileHandle = entry;
        const file = await fileHandle.getFile();
        const relativePath = basePath ? `${basePath}/${entry.name}` : entry.name;
        attachRelativePath(file, relativePath);
        files.push(file);
      }
    } else if (entry.kind === 'directory' && maxDepth > 0) {
      // Recurse into subdirectories (but not too deep)
      const subDirHandle = entry;
      const nextBasePath = basePath ? `${basePath}/${entry.name}` : entry.name;
      const subFiles = await collectModelFilesFromDirectory(subDirHandle, nextBasePath, maxDepth - 1);
      files.push(...subFiles);
    }
  }

  return files;
}

// ============================================================================
// File Input Fallback Implementation
// ============================================================================


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

      for (const file of modelFiles) {
        if (!file.webkitRelativePath) continue;
        const parts = file.webkitRelativePath.split('/').filter(Boolean);
        if (parts.length > 1) {
          attachRelativePath(file, parts.slice(1).join('/'));
        }
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


export function canStreamFile(file) {
  return typeof file.stream === 'function';
}


export function detectModelFormat(files) {
  for (const file of files) {
    const name = file.name.toLowerCase();
    if (name.endsWith('.gguf')) return 'gguf';
    if (name.endsWith('.safetensors')) return 'safetensors';
  }
  return 'unknown';
}

export default pickGGUFFile;
