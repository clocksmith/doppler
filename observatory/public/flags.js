// File-based feature flags. A flag turns a capability on or off; it never
// changes what counts as a valid result. That belongs to the versioned protocol.

const SCHEMA = 'doppler.observatory.flags/v1';

export function validateFlags(config) {
  if (config?.schema !== SCHEMA || !config.flags || typeof config.flags !== 'object') {
    throw new Error(`Flags file must use schema ${SCHEMA}.`);
  }
  for (const [name, flag] of Object.entries(config.flags)) {
    if (typeof flag?.enabled !== 'boolean') throw new Error(`Flag ${name} needs a boolean "enabled".`);
    if (flag.enforcedBy !== 'client' && flag.enforcedBy !== 'server') throw new Error(`Flag ${name} needs "enforcedBy": "client" or "server".`);
    if (typeof flag.description !== 'string' || !flag.description.trim()) throw new Error(`Flag ${name} needs a description.`);
    // A switch that guards publication or votes is decided by a server. A page
    // can hide a button; it cannot be the thing that permits the action.
    if (flag.enforcedBy === 'server' && (flag.previewOverride === true || flag.enabled === true)) {
      throw new Error(`Flag ${name} is server-enforced: this file cannot enable it or allow a preview of it.`);
    }
  }
  return config;
}

// `preview` holds flag names a visitor asked to preview (from ?preview=a,b).
// Only client-enforced flags that allow it can be previewed.
export function evaluateFlags(config, { preview = [] } = {}) {
  validateFlags(config);
  const asked = new Set(preview);
  return Object.fromEntries(Object.entries(config.flags).map(([name, flag]) => {
    const previewed = !flag.enabled && flag.enforcedBy === 'client' && flag.previewOverride === true && asked.has(name);
    return [name, { enabled: flag.enabled || previewed, source: previewed ? 'preview' : 'file', enforcedBy: flag.enforcedBy }];
  }));
}

export function previewNames(search) {
  return (new URLSearchParams(search).get('preview') ?? '').split(',').map((name) => name.trim()).filter(Boolean);
}

export async function loadFlags(url = new URL('./config/flags.json', import.meta.url), search = globalThis.location?.search ?? '') {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Flags file unavailable (HTTP ${response.status}).`);
  return evaluateFlags(await response.json(), { preview: previewNames(search) });
}
