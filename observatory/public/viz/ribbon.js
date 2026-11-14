// A probability ribbon: one band per listed candidate, as wide as its
// probability, and always a last band for every token that is not listed.

import { probabilityBands } from './scale.js';

const percent = (share) => `${(share * 100).toFixed(share >= 0.1 ? 0 : share >= 0.001 ? 1 : 2)}%`;

// candidates: { key, text, probability, state: 'taken' | 'run' | 'not-run' }.
export function ribbon(candidates, { label, onPick = null, unavailable = '' } = {}) {
  const strip = document.createElement('div');
  strip.className = 'site-ribbon';
  strip.setAttribute('role', 'group');
  if (label) strip.setAttribute('aria-label', label);
  for (const band of probabilityBands(candidates)) {
    const node = document.createElement(band.other ? 'span' : 'button');
    node.className = 'site-band';
    node.style.flexGrow = String(Math.max(band.share, 0.0001));
    node.dataset.state = band.other ? 'other' : band.state;
    const said = band.other ? 'tokens not listed' : band.text;
    node.textContent = `${said} ${percent(band.share)}`;
    if (band.other) node.title = 'Probability on tokens outside the list. They exist; they are not shown and cannot be continued from.';
    else {
      node.type = 'button';
      node.title = { taken: 'Taken by this run', run: 'Run: show this branch', 'not-run': unavailable || 'Not run: continue from here' }[band.state];
      node.disabled = band.state === 'taken' || (band.state === 'not-run' && Boolean(unavailable));
      if (onPick) node.addEventListener('click', () => onPick(band));
    }
    strip.append(node);
  }
  return strip;
}
