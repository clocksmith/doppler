import { log } from '../src/debug/index.js';

const DEFAULT_EXCLUDE_DIRS = ['.git', 'node_modules'];

async function collectEntries(dirHandle, basePath, entries, excludeDirs) {
  for await (const [name, handle] of dirHandle.entries()) {
    if (handle.kind === 'directory') {
      if (excludeDirs.includes(name)) continue;
      const nextBase = basePath ? `${basePath}/${name}` : name;
      await collectEntries(handle, nextBase, entries, excludeDirs);
    } else if (handle.kind === 'file') {
      const filePath = basePath ? `/${basePath}/${name}` : `/${name}`;
      entries.push({ path: filePath, handle });
    }
  }
}

export function createWorkspaceImporter(options = {}) {
  const excludeDirs = options.excludeDirs || DEFAULT_EXCLUDE_DIRS;
  const worker = new Worker(new URL('./workspace-import-worker.js', import.meta.url), { type: 'module' });
  let pending = null;

  worker.onmessage = (event) => {
    const data = event.data || {};
    if (!pending) return;
    if (data.type === 'progress') {
      pending.onProgress?.(data);
      return;
    }
    if (data.type === 'done') {
      const resolve = pending.resolve;
      pending = null;
      resolve(data);
      return;
    }
    if (data.type === 'error') {
      const reject = pending.reject;
      pending = null;
      reject(new Error(data.message || 'Import failed'));
    }
  };

  async function importDirectory(params = {}) {
    if (pending) {
      throw new Error('Import already running');
    }

    let dirHandle;
    try {
      dirHandle = await window.showDirectoryPicker();
    } catch (error) {
      if (error?.name === 'AbortError') return null;
      throw error;
    }

    const permission = await dirHandle.requestPermission({ mode: 'readwrite' });
    if (permission !== 'granted') {
      throw new Error('Permission denied');
    }

    const entries = [];
    await collectEntries(dirHandle, '', entries, excludeDirs);

    const workspaceId = params.workspaceId || dirHandle.name || 'default';

    const result = await new Promise((resolve, reject) => {
      pending = {
        resolve,
        reject,
        onProgress: options.onProgress,
      };
      worker.postMessage({
        type: 'import',
        workspaceId,
        rootDirName: options.rootDirName,
        backendType: options.backendType,
        entries,
      });
    });

    log.info('WorkspaceImporter', `Imported ${result.total || 0} files`);
    return result;
  }

  function terminate() {
    worker.terminate();
  }

  return { importDirectory, terminate };
}
