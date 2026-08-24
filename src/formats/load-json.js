const isNodeRuntime = typeof process !== 'undefined'
  && typeof process.versions === 'object'
  && typeof process.versions.node === 'string';

function nodeModule(specifier) {
  return `node:${specifier}`;
}

export async function loadJson(resourcePath, baseUrl = import.meta.url, errorPrefix = 'Failed to load JSON') {
  const resolved = new URL(resourcePath, baseUrl);
  if (isNodeRuntime && resolved.protocol === 'file:') {
    const fs = await import(nodeModule('fs/promises'));
    const { fileURLToPath } = await import(nodeModule('url'));
    const filePath = fileURLToPath(resolved);
    const raw = await fs.readFile(filePath, 'utf-8');
    try {
      return JSON.parse(raw);
    } catch (parseError) {
      throw new Error(`${errorPrefix}: JSON parse error in "${filePath}": ${parseError.message}`);
    }
  }

  const response = await fetch(resolved);
  if (!response.ok) {
    throw new Error(`${errorPrefix}: ${resourcePath}`);
  }
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch (parseError) {
    throw new Error(`${errorPrefix}: JSON parse error from "${resolved.href}": ${parseError.message}`);
  }
}
