import fs from 'node:fs/promises';

let fileFetchShimInstalled = false;

export function installNodeFileFetchShim() {
  if (fileFetchShimInstalled) return;
  if (typeof globalThis.fetch !== 'function') return;

  const originalFetch = globalThis.fetch.bind(globalThis);

  globalThis.fetch = async (input, init) => {
    const source = typeof input === 'string'
      ? input
      : input instanceof URL
        ? input.href
        : input?.url || '';

    let url;
    try {
      url = new URL(source, 'file://');
    } catch {
      return originalFetch(input, init);
    }

    if (url.protocol !== 'file:') {
      return originalFetch(input, init);
    }

    try {
      const body = await fs.readFile(url);
      return new Response(body, { status: 200 });
    } catch (error) {
      return new Response(String(error?.message || 'Not Found'), { status: 404 });
    }
  };

  fileFetchShimInstalled = true;
}
