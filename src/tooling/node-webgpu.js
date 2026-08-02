import { readFileSync } from 'node:fs';

const PROVIDER_CONTRACT_SPECIFIER = 'doe-gpu/node-webgpu';
const PROVIDER_CONTRACT_SCHEMA = 'doe.webgpu-provider/v1';
const DEFAULT_PROVIDER_CONFIG_URL = new URL(
  './node-webgpu-provider.v1.json',
  import.meta.url,
);
const WEBGPU_GLOBAL_BINDINGS = Object.freeze({
  GPUBufferUsage: 'globals.GPUBufferUsage',
  GPUShaderStage: 'globals.GPUShaderStage',
  GPUMapMode: 'globals.GPUMapMode',
  GPUTextureUsage: 'globals.GPUTextureUsage',
});

let ownedProviderSession = null;

function readDefaultProviderOptions() {
  const parsed = JSON.parse(readFileSync(DEFAULT_PROVIDER_CONFIG_URL, 'utf8'));
  if (parsed?.schema !== PROVIDER_CONTRACT_SCHEMA) {
    throw new DopplerNodeWebGPUError(
      'DOPPLER_PROVIDER_CONFIG_INVALID',
      `default Node WebGPU provider config must declare schema "${PROVIDER_CONTRACT_SCHEMA}".`,
      { stage: 'configuration' },
    );
  }
  const { schema: _schema, ...providerOptions } = parsed;
  return providerOptions;
}

function environmentProviderOptions(defaultOptions) {
  const moduleSpecifier = typeof process.env.DOPPLER_NODE_WEBGPU_MODULE === 'string'
    ? process.env.DOPPLER_NODE_WEBGPU_MODULE.trim()
    : '';
  if (!moduleSpecifier) return defaultOptions;

  const rawArgs = typeof process.env.DOPPLER_NODE_WEBGPU_PROVIDER_ARGS === 'string'
    ? process.env.DOPPLER_NODE_WEBGPU_PROVIDER_ARGS.trim()
    : '';
  const createArgs = rawArgs
    ? rawArgs.split(',').map((value) => value.trim()).filter(Boolean)
    : ['enable-dawn-features=allow_unsafe_apis'];
  return {
    providers: [{
      id: moduleSpecifier,
      kind: 'module',
      module: moduleSpecifier,
      gpu: { kind: 'factory', path: 'create', args: [createArgs] },
      globals: { ...WEBGPU_GLOBAL_BINDINGS },
    }],
    adapterOptions: defaultOptions.adapterOptions,
    globals: defaultOptions.globals,
  };
}

async function loadProviderContract(specifier = PROVIDER_CONTRACT_SPECIFIER) {
  let contract;
  try {
    contract = await import(specifier);
  } catch (cause) {
    throw new DopplerNodeWebGPUError(
      'DOPPLER_PROVIDER_CONTRACT_UNAVAILABLE',
      `Node WebGPU provider-v1 contract "${specifier}" is unavailable: ${cause?.message || String(cause)}`,
      { stage: 'contract.import', cause },
    );
  }
  if (
    contract.NODE_WEBGPU_PROVIDER_SCHEMA !== PROVIDER_CONTRACT_SCHEMA
    || typeof contract.openNodeWebGPU !== 'function'
  ) {
    throw new DopplerNodeWebGPUError(
      'DOPPLER_PROVIDER_CONTRACT_INVALID',
      `Node WebGPU provider contract "${specifier}" does not implement ${PROVIDER_CONTRACT_SCHEMA}.`,
      { stage: 'contract.validate' },
    );
  }
  return contract;
}

function providerFromSpecifier(providerSpecifier, options = {}) {
  if (typeof providerSpecifier !== 'string' || providerSpecifier.trim().length === 0) {
    throw new DopplerNodeWebGPUError(
      'DOPPLER_PROVIDER_CONFIG_INVALID',
      'providerSpecifier must be a non-empty string.',
      { stage: 'configuration' },
    );
  }
  const createArgs = options.createArgs ?? ['enable-dawn-features=allow_unsafe_apis'];
  if (!Array.isArray(createArgs)) {
    throw new DopplerNodeWebGPUError(
      'DOPPLER_PROVIDER_CONFIG_INVALID',
      'provider createArgs must be an array.',
      { stage: 'configuration' },
    );
  }
  return {
    id: options.id ?? providerSpecifier,
    kind: 'module',
    module: providerSpecifier,
    gpu: options.gpu ?? { kind: 'factory', path: 'create', args: [createArgs] },
    globals: options.globals ?? { ...WEBGPU_GLOBAL_BINDINGS },
  };
}

function resultFromSession(session) {
  return {
    ok: true,
    provider: session.receipt.selectedProviderId,
    detail: null,
    module: session.module,
    session,
    receipt: session.receipt,
  };
}

export class DopplerNodeWebGPUError extends Error {
  constructor(code, message, options = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = 'DopplerNodeWebGPUError';
    this.code = code;
    this.stage = options.stage ?? 'provider';
    this.receipt = options.receipt ?? null;
  }
}

export async function openNodeWebGPU(providerOptions, options = {}) {
  if (ownedProviderSession !== null) {
    throw new DopplerNodeWebGPUError(
      'DOPPLER_PROVIDER_ALREADY_ACTIVE',
      'a Doppler-owned Node WebGPU provider session is already active; release it before selecting another provider.',
      { stage: 'lifecycle' },
    );
  }
  const contract = await loadProviderContract(options.providerContractModule);
  const session = await contract.openNodeWebGPU(providerOptions);
  ownedProviderSession = session;
  return session;
}

export async function bootstrapNodeWebGPUProvider(providerSpecifier, options = {}) {
  const provider = options.provider ?? providerFromSpecifier(providerSpecifier, options);
  const session = await openNodeWebGPU({
    providers: [provider],
    adapterOptions: options.adapterOptions ?? null,
    globals: { mode: options.globalMode ?? 'replace' },
  }, options);
  return resultFromSession(session);
}

export async function bootstrapNodeWebGPU(options = {}) {
  if (ownedProviderSession !== null) {
    return resultFromSession(ownedProviderSession);
  }
  let providerOptions;
  try {
    providerOptions = options.providerOptions
      ?? environmentProviderOptions(readDefaultProviderOptions());
    const session = await openNodeWebGPU(providerOptions, options);
    return resultFromSession(session);
  } catch (error) {
    const receipt = error?.receipt ?? null;
    const lastAttempt = Array.isArray(receipt?.attempts)
      ? receipt.attempts[receipt.attempts.length - 1]
      : null;
    return {
      ok: false,
      provider: lastAttempt?.providerId ?? null,
      detail: error?.message || String(error),
      module: null,
      session: null,
      receipt,
      error,
    };
  }
}

export async function releaseNodeWebGPU() {
  const session = ownedProviderSession;
  ownedProviderSession = null;
  if (!session) {
    return { released: false, provider: null, reason: 'not-owned', receipt: null };
  }
  const provider = session.receipt?.selectedProviderId ?? null;
  try {
    await session.close();
    return { released: true, provider, reason: null, receipt: session.receipt };
  } catch (cause) {
    throw new DopplerNodeWebGPUError(
      'DOPPLER_PROVIDER_RELEASE_FAILED',
      `failed to release Node WebGPU provider "${provider}": ${cause?.message || String(cause)}`,
      { stage: 'lifecycle.release', cause, receipt: session.receipt },
    );
  }
}
