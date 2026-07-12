import { sha256Hex } from '../../src/utils/sha256.js';

const DEFAULT_BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-dawn-features=allow_unsafe_apis',
  '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE',
  '--use-angle=vulkan',
  '--disable-vulkan-surface',
]);

function normalizeMessages(messages) {
  return messages.map((message) => ({
    type: message.type,
    message: message.message,
    lineNum: message.lineNum,
    linePos: message.linePos,
    offset: message.offset,
    length: message.length,
  }));
}

export async function createWgslBrowserVerifier(options = {}) {
  const { chromium } = await import('playwright');
  const browserArgs = Array.isArray(options.browserArgs)
    ? options.browserArgs.map(String)
    : [...DEFAULT_BROWSER_ARGS];
  const browser = await chromium.launch({
    headless: options.headless !== false,
    args: browserArgs,
  });
  const page = await browser.newPage();
  await page.route('https://wgsl-verifier.invalid/**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/html',
      body: '<!doctype html><meta charset="utf-8"><title>WGSL verifier</title>',
    });
  });
  await page.goto('https://wgsl-verifier.invalid/');
  const deviceInfo = await page.evaluate(async (request) => {
    if (!navigator.gpu) {
      throw new Error('webgpu_unavailable');
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: request.powerPreference,
    });
    if (!adapter) {
      throw new Error('webgpu_adapter_unavailable');
    }
    const availableFeatures = [...adapter.features].sort();
    const missingFeatures = request.requiredFeatures.filter((feature) => (
      !adapter.features.has(feature)
    ));
    if (missingFeatures.length > 0) {
      throw new Error(`webgpu_required_features_missing:${missingFeatures.join(',')}`);
    }
    const device = await adapter.requestDevice({
      requiredFeatures: request.requiredFeatures,
    });
    globalThis.__wgslVerifierDevice = device;
    const adapterInfo = adapter.info || {};
    return {
      vendor: adapterInfo.vendor || null,
      architecture: adapterInfo.architecture || null,
      device: adapterInfo.device || null,
      description: adapterInfo.description || null,
      availableFeatures,
      requiredFeatures: request.requiredFeatures,
    };
  }, {
    powerPreference: options.powerPreference || 'high-performance',
    requiredFeatures: Array.isArray(options.requiredFeatures)
      ? options.requiredFeatures.map(String)
      : ['shader-f16', 'subgroups'],
  });
  const requiredVendor = String(options.requiredVendor || '').trim().toLowerCase();
  if (requiredVendor && String(deviceInfo.vendor || '').toLowerCase() !== requiredVendor) {
    await browser.close();
    throw new Error(
      `webgpu_adapter_vendor_mismatch: expected ${requiredVendor}, got ${deviceInfo.vendor || 'unknown'}`
    );
  }

  return {
    deviceInfo,
    browserArgs,
    async compile(entries) {
      const normalizedEntries = entries.map((entry, index) => ({
        id: String(entry.id || `shader-${index + 1}`),
        code: String(entry.code || ''),
      }));
      const compiled = await page.evaluate(async (shaders) => {
        const device = globalThis.__wgslVerifierDevice;
        if (!device) throw new Error('webgpu_verifier_device_missing');
        const results = [];
        for (const shader of shaders) {
          const module = device.createShaderModule({
            code: shader.code,
            label: shader.id,
          });
          const info = await module.getCompilationInfo();
          results.push({
            id: shader.id,
            messages: [...info.messages].map((message) => ({
              type: message.type,
              message: message.message,
              lineNum: message.lineNum,
              linePos: message.linePos,
              offset: message.offset,
              length: message.length,
            })),
          });
        }
        return results;
      }, normalizedEntries);
      return compiled.map((result, index) => {
        const messages = normalizeMessages(result.messages);
        const errors = messages.filter((message) => message.type === 'error');
        return {
          id: result.id,
          sourceSha256: sha256Hex(normalizedEntries[index].code),
          passed: errors.length === 0,
          messages,
          errorCount: errors.length,
        };
      });
    },
    async close() {
      await browser.close();
    },
  };
}

export async function verifyWgslCompilation(entries, options = {}) {
  const verifier = await createWgslBrowserVerifier(options);
  try {
    return {
      deviceInfo: verifier.deviceInfo,
      browserArgs: verifier.browserArgs,
      results: await verifier.compile(entries),
    };
  } finally {
    await verifier.close();
  }
}
