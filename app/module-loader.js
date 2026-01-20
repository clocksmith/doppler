export async function loadWorkspaceModule(vfs, path, options = {}) {
  if (!vfs || typeof vfs.readText !== 'function') {
    throw new Error('VFS not available');
  }

  const force = options.force === true;
  const code = await vfs.readText(path);
  if (code === null || code === undefined) {
    throw new Error(`Module not found: ${path}`);
  }

  const cached = moduleCache.get(path);
  if (!force && cached && cached.code === code) {
    return cached.module;
  }

  const blob = new Blob([code], { type: 'text/javascript' });
  const url = URL.createObjectURL(blob);
  try {
    const mod = await import(url);
    moduleCache.set(path, { code, module: mod });
    return mod;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export function clearWorkspaceModuleCache(path) {
  if (!path) {
    moduleCache.clear();
    return { cleared: 'all' };
  }
  const existed = moduleCache.delete(path);
  return { cleared: path, existed };
}

const moduleCache = new Map();
