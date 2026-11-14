// What a run did not record is drawn as a gap that says so. Nothing is filled
// in. Where the runtime can record the missing part, the gap offers to.

import { provenanceMark } from './provenance.js';

// A selective-capture request for Doppler's `capture` option: only the named
// roles, layers and token range. Empty or absent parts are left to the tier.
export function captureRequest({ roles = null, layers = null, tokens = null } = {}) {
  const request = {};
  if (roles !== null) {
    if (!Array.isArray(roles) || roles.some((role) => typeof role !== 'string' || !role)) throw new Error('Capture roles must be a list of role names.');
    request.roles = [...new Set(roles)];
  }
  if (layers !== null) {
    if (!Array.isArray(layers) || layers.some((layer) => !Number.isInteger(layer) || layer < 0)) throw new Error('Capture layers must be whole numbers from 0.');
    request.layers = [...new Set(layers)].sort((a, b) => a - b);
  }
  if (tokens !== null) {
    if (!Number.isInteger(tokens.from) || !Number.isInteger(tokens.to) || tokens.from < 0 || tokens.to < tokens.from) throw new Error('A capture token range needs whole numbers with from <= to.');
    request.tokens = { from: tokens.from, to: tokens.to };
  }
  return request;
}

// `action` is { label, run, unavailable }: `unavailable` is the reason it cannot run here.
// `kind` is the provenance of the hole: `missing` (the run did not record it) or
// `unrun` (it could be computed and has not been).
export function gap({ what, action = null, kind = 'missing' }) {
  const box = document.createElement('div');
  box.className = 'site-gap';
  const text = document.createElement('span');
  text.textContent = what;
  box.append(provenanceMark(kind), text);
  if (action) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn btn-small';
    button.textContent = action.label;
    button.disabled = Boolean(action.unavailable);
    if (action.unavailable) button.title = action.unavailable;
    button.addEventListener('click', action.run);
    box.append(button);
    if (action.unavailable) { const why = document.createElement('span'); why.className = 'demo-note'; why.textContent = action.unavailable; box.append(why); }
  }
  return box;
}
