export const NODE_WEBGPU_PROVIDER_SCHEMA = 'doe.webgpu-provider/v1';

let lastProviderOptions = null;

export async function openNodeWebGPU(providerOptions) {
  lastProviderOptions = structuredClone(providerOptions);
  const selectedProviderId = providerOptions.providers[0].id;
  const receipt = {
    schema: 'doe.webgpu-provider-receipt/v1',
    contract: NODE_WEBGPU_PROVIDER_SCHEMA,
    providers: structuredClone(providerOptions.providers),
    providerOrder: providerOptions.providers.map((provider) => provider.id),
    adapterOptions: structuredClone(providerOptions.adapterOptions),
    globals: {
      mode: providerOptions.globals.mode,
      installed: [],
      restored: false,
    },
    attempts: [{
      providerId: selectedProviderId,
      kind: providerOptions.providers[0].kind,
      module: providerOptions.providers[0].module ?? null,
      ok: true,
      stage: 'complete',
      code: null,
      detail: null,
    }],
    selectedProviderId,
    ok: true,
  };
  return {
    gpu: { requestAdapter: async () => ({}) },
    adapter: {},
    module: { fixture: true },
    receipt,
    async close() {
      receipt.globals.restored = true;
    },
  };
}

export function getLastProviderOptions() {
  return structuredClone(lastProviderOptions);
}
