// A layer axis: one stop per layer, shaded by a value from 0 to 1, with one
// stop selected. Views that walk the layers share it.

export function track({ count, selected = null, strength = () => 0, title = (layer) => `Layer ${layer + 1}`, onSelect, label = 'Layers' }) {
  const axis = document.createElement('div');
  axis.className = 'site-track';
  axis.setAttribute('role', 'group');
  axis.setAttribute('aria-label', label);
  for (let layer = 0; layer < count; layer++) {
    const stop = document.createElement('button');
    stop.type = 'button';
    stop.className = 'site-stop';
    stop.textContent = String(layer + 1);
    stop.title = title(layer);
    stop.setAttribute('aria-pressed', String(layer === selected));
    const value = strength(layer);
    // No value for a layer is a gap, not a zero.
    if (value === null) stop.dataset.missing = 'true';
    else stop.style.setProperty('--site-strength', value.toFixed(2));
    stop.addEventListener('click', () => onSelect(layer === selected ? null : layer));
    axis.append(stop);
  }
  return axis;
}
