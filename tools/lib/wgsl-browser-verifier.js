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

function requirePositiveInteger(value, fallback, label) {
  const parsed = value == null ? fallback : Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be an integer >= 1.`);
  }
  return parsed;
}

export async function settleWithTimeout(operation, timeoutMs) {
  let timeoutId;
  const operationResult = Promise.resolve()
    .then(operation)
    .then(
      (value) => ({ status: 'fulfilled', value }),
      (error) => ({ status: 'rejected', error })
    );
  const timeoutResult = new Promise((resolve) => {
    timeoutId = setTimeout(() => resolve({ status: 'timed_out' }), timeoutMs);
  });
  const result = await Promise.race([operationResult, timeoutResult]);
  clearTimeout(timeoutId);
  return result;
}

export async function createWgslBrowserVerifier(options = {}) {
  const { chromium } = await import('playwright');
  const browserArgs = Array.isArray(options.browserArgs)
    ? options.browserArgs.map(String)
    : [...DEFAULT_BROWSER_ARGS];
  const compilationTimeoutMs = requirePositiveInteger(
    options.compilationTimeoutMs,
    10000,
    'compilationTimeoutMs'
  );
  const progressEvery = requirePositiveInteger(options.progressEvery, 100, 'progressEvery');
  const deviceRequest = {
    powerPreference: options.powerPreference || 'high-performance',
    requiredFeatures: Array.isArray(options.requiredFeatures)
      ? options.requiredFeatures.map(String)
      : ['shader-f16', 'subgroups'],
  };
  const requiredVendor = String(options.requiredVendor || '').trim().toLowerCase();

  let browser = null;
  let page = null;
  async function openSession() {
    browser = await chromium.launch({
      headless: options.headless !== false,
      args: browserArgs,
    });
    page = await browser.newPage();
    await page.route('https://wgsl-verifier.invalid/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/html',
        body: '<!doctype html><meta charset="utf-8"><title>WGSL verifier</title>',
      });
    });
    await page.goto('https://wgsl-verifier.invalid/');
    const info = await page.evaluate(async (request) => {
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
    }, deviceRequest);
    if (requiredVendor && String(info.vendor || '').toLowerCase() !== requiredVendor) {
      await browser.close();
      browser = null;
      page = null;
      throw new Error(
        `webgpu_adapter_vendor_mismatch: expected ${requiredVendor}, got ${info.vendor || 'unknown'}`
      );
    }
    return info;
  }

  const deviceInfo = await openSession();
  async function resetSession() {
    if (browser) await browser.close().catch(() => {});
    browser = null;
    page = null;
    const replacementInfo = await openSession();
    if (JSON.stringify(replacementInfo) !== JSON.stringify(deviceInfo)) {
      throw new Error('webgpu_verifier_device_changed_after_reset');
    }
  }

  return {
    deviceInfo,
    browserArgs,
    compilationTimeoutMs,
    async compile(entries) {
      const normalizedEntries = entries.map((entry, index) => ({
        id: String(entry.id || `shader-${index + 1}`),
        code: String(entry.code || ''),
      }));
      const compiled = [];
      for (const [index, shader] of normalizedEntries.entries()) {
        const outcome = await settleWithTimeout(
          () => page.evaluate(async (input) => {
            const device = globalThis.__wgslVerifierDevice;
            if (!device) throw new Error('webgpu_verifier_device_missing');
            const module = device.createShaderModule({
              code: input.code,
              label: input.id,
            });
            const info = await module.getCompilationInfo();
            return {
              id: input.id,
              messages: [...info.messages].map((message) => ({
                type: message.type,
                message: message.message,
                lineNum: message.lineNum,
                linePos: message.linePos,
                offset: message.offset,
                length: message.length,
              })),
            };
          }, shader),
          compilationTimeoutMs
        );
        if (outcome.status === 'fulfilled') {
          compiled.push(outcome.value);
        } else {
          const reason = outcome.status === 'timed_out'
            ? `WGSL compilation timed out after ${compilationTimeoutMs}ms.`
            : `WGSL verifier exception: ${outcome.error?.message || String(outcome.error)}`;
          compiled.push({
            id: shader.id,
            messages: [{
              type: 'error',
              message: reason,
              lineNum: 0,
              linePos: 0,
              offset: 0,
              length: 0,
            }],
          });
          await resetSession();
        }
        if ((index + 1) % progressEvery === 0 || index + 1 === normalizedEntries.length) {
          console.error(`[wgsl-verifier] compiled ${index + 1}/${normalizedEntries.length}`);
        }
      }
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
      if (browser) await browser.close();
      browser = null;
      page = null;
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
