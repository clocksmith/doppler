// A time strip: one segment per step, as wide as the time it took. Selecting a
// segment unfolds it; the view decides what is inside.

export function timeline(segments, { selected = null, onPick, label = 'Time per step' }) {
  const strip = document.createElement('div');
  strip.className = 'site-timeline';
  strip.setAttribute('role', 'group');
  strip.setAttribute('aria-label', label);
  for (const segment of segments) {
    if (!(segment.ms >= 0)) throw new Error('A timeline segment needs a measured time.');
    const node = document.createElement('button');
    node.type = 'button';
    node.className = 'site-segment';
    node.style.flexGrow = String(Math.max(segment.ms, 0.0001));
    node.title = segment.title;
    node.textContent = segment.label;
    node.setAttribute('aria-pressed', String(segment.key === selected));
    node.setAttribute('aria-expanded', String(segment.key === selected));
    node.addEventListener('click', () => onPick(segment.key === selected ? null : segment.key));
    strip.append(node);
  }
  return strip;
}
