// Builds an element. Shared by the views so each does not carry its own copy.
export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === 'on') for (const [event, handler] of Object.entries(value)) node.addEventListener(event, handler);
    else if (key === 'data') Object.assign(node.dataset, value);
    else if (key === 'attrs') for (const [name, text] of Object.entries(value)) node.setAttribute(name, text);
    else if (value !== null && value !== undefined && value !== false) node[key] = value;
  }
  node.append(...children.flat(Infinity).filter((child) => child !== null && child !== undefined && child !== false));
  return node;
}

export const $ = (id) => document.getElementById(id);
export const percent = (value) => (Number.isFinite(value) ? `${(value * 100).toFixed(value >= 0.1 ? 0 : value >= 0.001 ? 1 : 2)}%` : 'not recorded');
export const ms = (value) => (Number.isFinite(value) ? `${value.toFixed(value < 10 ? 2 : 1)} ms` : 'not measured');
export const visible = (text) => (text === null ? '(text not recorded)' : text === '' ? '(empty)' : text.replace(/\n/g, '⏎'));
export const where = (run) => [[run.environment.adapter?.vendor, run.environment.adapter?.architecture].filter(Boolean).join(' '), run.environment.runtimeVersion ? `Doppler ${run.environment.runtimeVersion}${run.environment.runtimeChannel === 'preview' ? ' preview' : ''}` : null].filter(Boolean).join(' · ') || 'not recorded';
