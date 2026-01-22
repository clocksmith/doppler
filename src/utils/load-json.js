const isNodeRuntime = typeof process !== 'undefined'
  && typeof process.versions === 'object'
  && typeof process.versions.node === 'string';

export async function loadJson(resourcePath, baseUrl = import.meta.url, errorPrefix = 'Failed to load JSON') {
  const resolved = new URL(resourcePath, baseUrl);
  if (isNodeRuntime && resolved.protocol === 'file:') {
    const fs = await import('node:fs/promises');
    const { fileURLToPath } = await import('node:url');
    const raw = await fs.readFile(fileURLToPath(resolved), 'utf-8');
    return JSON.parse(raw);
  }

  const response = await fetch(resolved);
  if (!response.ok) {
    throw new Error(`${errorPrefix}: ${resourcePath}`);
  }
  return response.json();
}
