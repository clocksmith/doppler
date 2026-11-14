// Which Doppler runtime a visit uses. The hosted runtime is the default. A
// preview runtime is a newer, unqualified build: it is offered only when its
// switch is on, used only when the visitor also asks for it, and every result
// made with it says so.

const SCHEMA = 'doppler.observatory.runtimes/v1';

export function validateRuntimes(config) {
  if (config?.schema !== SCHEMA || !config.runtimes || typeof config.runtimes !== 'object') throw new Error(`Runtimes file must use schema ${SCHEMA}.`);
  for (const [id, runtime] of Object.entries(config.runtimes)) {
    if (typeof runtime?.label !== 'string' || !runtime.label) throw new Error(`Runtime ${id} needs a label.`);
    // Only addresses on this site: a runtime is code the page will run.
    for (const field of ['module', 'release']) if (!/^(\/|\.\/)[\w./-]+$/.test(runtime[field] ?? '')) throw new Error(`Runtime ${id} needs a same-site ${field} address.`);
    if (typeof runtime.preview !== 'boolean') throw new Error(`Runtime ${id} must say whether it is a preview.`);
    // A preview is never the silent default: it sits behind a switch and carries its reason.
    if (runtime.preview && (typeof runtime.flag !== 'string' || typeof runtime.notice !== 'string' || !runtime.notice)) throw new Error(`Preview runtime ${id} needs a switch and a notice.`);
  }
  const fallback = config.runtimes[config.default];
  if (!fallback || fallback.preview || fallback.flag) throw new Error('The default runtime must exist, be unconditional and not be a preview.');
  return config;
}

// The runtime for this visit, and the ones a visitor may switch between.
// `?runtime=<id>` is honored only for a runtime whose switch is on.
export function chooseRuntime(config, flags, search = '') {
  validateRuntimes(config);
  const offered = Object.entries(config.runtimes).filter(([, runtime]) => !runtime.flag || flags?.[runtime.flag]?.enabled === true).map(([id]) => id);
  const asked = new URLSearchParams(search).get('runtime');
  const id = asked && offered.includes(asked) ? asked : config.default;
  return { id, ...config.runtimes[id], offered: offered.map((item) => ({ id: item, label: config.runtimes[item].label, preview: config.runtimes[item].preview })) };
}

// "Doppler 0.7.0 (b9116524) · preview, not qualified"
export function describeRuntime(runtime) {
  const version = runtime.release?.label ?? runtime.release?.version ?? runtime.version ?? null;
  const commit = runtime.release?.sourceCommit ? ` (${runtime.release.sourceCommit.slice(0, 8)})` : '';
  return `${version ? `Doppler ${version}${commit}` : 'Doppler, version not reported'}${runtime.preview ? ' · preview, not qualified' : ''}`;
}

export async function loadRuntimeChoice(flags, search = globalThis.location?.search ?? '') {
  const response = await fetch(new URL('./config/runtimes.json', import.meta.url), { cache: 'no-store' });
  if (!response.ok) throw new Error(`Runtimes file unavailable (HTTP ${response.status}).`);
  const choice = chooseRuntime(await response.json(), flags, search);
  const base = new URL('./', import.meta.url);
  const runtime = { ...choice, moduleUrl: new URL(choice.module, base).href, release: null };
  try {
    const release = await fetch(new URL(choice.release, base), { cache: 'no-store' });
    if (release.ok) {
      const body = await release.json();
      runtime.release = {
        sourceCommit: body.runtime?.sourceCommit ?? null,
        version: body.runtime?.version ?? null,
        label: body.runtime?.label ?? null,
        uncommittedSourceChanges: body.runtime?.uncommittedSourceChanges ?? null,
      };
    }
  } catch { /* a runtime served without a release file reports only its own version */ }
  return runtime;
}

// A select that reloads the page on the chosen runtime, shown only when there is a choice.
export function runtimeSwitch(runtime) {
  if (runtime.offered.length < 2) return null;
  const select = document.createElement('select');
  select.id = 'runtime-switch';
  select.setAttribute('aria-label', 'Runtime');
  for (const item of runtime.offered) {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = `${item.label}${item.preview ? ' (not qualified)' : ''}`;
    option.selected = item.id === runtime.id;
    select.append(option);
  }
  select.addEventListener('change', () => {
    const url = new URL(location.href);
    url.searchParams.set('runtime', select.value);
    location.href = url.href;
  });
  return select;
}
