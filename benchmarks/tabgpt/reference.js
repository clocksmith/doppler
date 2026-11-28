export const tabgptRevision = '7e0387500f0f64bddf3e68c341508f3f4454cc81';

async function get(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('Could not fetch ' + url + ': HTTP ' + response.status);
  return response;
}

async function readTokens(url, expected) {
  const bytes = await (await get(url)).arrayBuffer();
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const digest = Array.from(hash, (n) => n.toString(16).padStart(2, '0')).join('');
  if (bytes.byteLength !== expected.tokens * 2 || digest !== expected.sha256) {
    throw new Error('TabGPT token stream failed provenance check: ' + url);
  }
  return new Uint16Array(bytes);
}

// Both runners reuse the original sampler with identical deterministic offsets.
function batchRng(seed) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new RangeError('The batch seed must be an unsigned 32-bit integer');
  }
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

async function preloadModule(url, hexDigest) {
  const bytes = hexDigest.match(/../g).map(hex => Number.parseInt(hex, 16));
  const link = document.createElement('link');
  link.rel = 'modulepreload';
  link.href = url;
  link.crossOrigin = 'anonymous';
  link.integrity = 'sha256-' + btoa(String.fromCharCode(...bytes));
  try {
    await new Promise((resolve, reject) => {
      link.onload = resolve;
      link.onerror = () => reject(new Error('TabGPT module failed integrity check: ' + url));
      document.head.append(link);
    });
  } finally {
    link.remove();
  }
}

export async function loadReference(assetBase) {
  const root = new URL(assetBase.replace(/\/?$/, '/'), location.href);
  const url = (name) => new URL(name, root).href;
  const provenance = await (await get(url('manifest.json'))).json();
  if (provenance.revision !== tabgptRevision) throw new Error('Unexpected TabGPT reference revision');
  // Verify the shared kernel module before its model dependency graph is loaded.
  await preloadModule(url('kernels.js'), provenance.modules.kernels.javascriptSha256);
  await Promise.all(['model', 'data'].map(name => preloadModule(url(name + '.js'), provenance.modules[name].javascriptSha256)));
  const [{ GPT }, { setF16 }, { sampleBatch }, train, val] = await Promise.all([
    import(url('model.js')), import(url('kernels.js')), import(url('data.js')),
    readTokens(url('train.bin'), provenance.splits.train),
    readTokens(url('val.bin'), provenance.splits.val),
  ]);
  const streams = { train, val };
  return {
    provenance,
    createModel(device, config) {
      setF16(config.f16);
      return new GPT(device, config);
    },
    batch(split, seed, B, T) {
      const data = streams[split];
      if (!(data instanceof Uint16Array)) throw new RangeError('Unknown TabGPT split: ' + split);
      if (!Number.isSafeInteger(B) || B < 1 || !Number.isSafeInteger(T) || T < 1 || T >= data.length - 1) {
        throw new RangeError('Invalid TabGPT batch dimensions');
      }
      return sampleBatch(data, data.length, B, T, batchRng(seed));
    },
  };
}
