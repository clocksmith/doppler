// What the pages of this site share: the links between them, and the two
// labels that say how a run was made.

const PAGES = [
  // The front page: ask, watch the answer form, look inside it, branch, compare runs.
  { id: 'explore', label: 'Explore', href: './', what: 'Ask a model and explore how the answer formed' },
  { id: 'run', label: 'Benchmark', href: './benchmark.html', what: 'Benchmark this device and compare results' },
];

// A link to another page that keeps a visitor's preview choices, and adds the
// preview a page needs when its switch is off.
export function pageHref(id, flags, search = globalThis.location?.search ?? '') {
  const page = PAGES.find((item) => item.id === id);
  const asked = new Set((new URLSearchParams(search).get('preview') ?? '').split(',').map((name) => name.trim()).filter(Boolean));
  if (page.flag && flags?.[page.flag] && !flags[page.flag].enabled) asked.add(page.flag);
  const query = new URLSearchParams();
  if (asked.size) query.set('preview', [...asked].join(','));
  // A visitor testing the preview runtime keeps it from page to page.
  const runtime = new URLSearchParams(search).get('runtime');
  if (runtime && /^[a-z-]+$/.test(runtime)) query.set('runtime', runtime);
  const text = query.toString().replace(/%2C/g, ',');
  return text ? `${page.href}?${text}` : page.href;
}

// Puts this site's pages at the front of the header's links (the demo's
// `.app-links`, which every page has), and marks the current one.
export function mountNav(current, flags, search) {
  const nav = document.querySelector('nav.app-links');
  if (!nav) return;
  for (const old of nav.querySelectorAll('a[data-site-page]')) old.remove();
  nav.prepend(...PAGES.map((page) => {
    const link = document.createElement('a');
    link.href = pageHref(page.id, flags, search);
    link.dataset.sitePage = page.id;
    const preview = page.flag && flags?.[page.flag]?.source === 'preview';
    const off = page.flag && flags?.[page.flag] && !flags[page.flag].enabled;
    link.textContent = `${page.label}${preview || off ? ' (preview)' : ''}`;
    link.title = page.what;
    if (page.id === current) link.setAttribute('aria-current', 'page');
    return link;
  }));
}

// The two ways a run is made here, and the one label each carries on every page.
// A benchmark measurement ran on the tier that records token IDs and wall time
// only. An instrumented recording ran with inspection, which adds work.
export const MODES = {
  benchmark: { label: 'Benchmark measurement', what: 'Run without inspection: token IDs and wall time only. Its timings are speed measurements.', filled: true },
  instrumented: { label: 'Instrumented recording', what: 'Recorded with inspection, which adds measurement work. Its timings are diagnostic, not speed measurements.', filled: false },
};

export function modeOf(run) {
  return run?.timing?.representative ? 'benchmark' : 'instrumented';
}

export function modeBadge(mode) {
  const { label, what, filled } = MODES[mode];
  const badge = document.createElement('span');
  badge.className = filled ? 'badge badge-filled' : 'badge';
  badge.dataset.mode = mode;
  badge.title = what;
  badge.textContent = label;
  return badge;
}

// Makes a table of records readable on a phone. There, site.css stacks each row
// of a `.site-cards` table into a card of labelled values, so each cell has to
// carry its column's heading. The roles keep it a table for assistive
// technology once the browser no longer lays it out as one. A heading marked
// data-card="title" leads its card; data-card="plain" is shown without a label.
export function cardTable(table) {
  table.classList.add('site-cards');
  table.setAttribute('role', 'table');
  for (const group of table.querySelectorAll('thead, tbody')) group.setAttribute('role', 'rowgroup');
  const heads = [...table.tHead.rows[0].cells];
  for (const head of heads) head.setAttribute('role', 'columnheader');
  for (const row of table.rows) {
    row.setAttribute('role', 'row');
    if (row.parentNode === table.tHead) continue;
    [...row.cells].forEach((cell, index) => {
      cell.setAttribute('role', 'cell');
      const head = heads[index];
      if (head?.dataset.card) cell.dataset.card = head.dataset.card;
      else if (head) cell.dataset.label = head.textContent.trim();
      // A cell with nothing in it is left out of its card.
      if (!cell.textContent.trim() && !cell.firstElementChild) { cell.replaceChildren(); return; }
      if (cell.childNodes.length === 1 && cell.firstElementChild?.classList.contains('site-value')) return;
      const value = document.createElement('div');
      value.className = 'site-value';
      value.append(...cell.childNodes);
      cell.append(value);
    });
  }
}

// A bar for a share between 0 and 1, drawn like the demo's candidate bars.
export function bar(share) {
  const track = document.createElement('span');
  track.className = 'site-bar';
  const fill = document.createElement('span');
  fill.style.width = `${Math.max(0, Math.min(1, Number.isFinite(share) ? share : 0)) * 100}%`;
  track.append(fill);
  return track;
}

// A saved run may be compressed (the kind that carries its grids is tens of
// megabytes otherwise). Reads either kind as text, by the file's first bytes.
export async function readSavedText(file) {
  const head = new Uint8Array(await file.slice(0, 2).arrayBuffer());
  if (head[0] !== 0x1f || head[1] !== 0x8b) return file.text();
  if (typeof DecompressionStream !== 'function') throw new Error('This browser cannot open a compressed run file.');
  return new Response(file.stream().pipeThrough(new DecompressionStream('gzip'))).text();
}

// Compresses text in pieces. The whole text is never handed to the browser as
// one stream: that failed intermittently for files of tens of megabytes.
async function gzip(text) {
  const stream = new CompressionStream('gzip');
  const writer = stream.writable.getWriter();
  const reader = stream.readable.getReader();
  const parts = [];
  const collected = (async () => { for (;;) { const { done, value } = await reader.read(); if (done) return; parts.push(value); } })();
  const encoder = new TextEncoder();
  const PIECE = 1 << 20;
  for (let start = 0; start < text.length; start += PIECE) await writer.write(encoder.encode(text.slice(start, start + PIECE)));
  await writer.close();
  await collected;
  return new Blob(parts, { type: 'application/gzip' });
}

// Offers text to the visitor as a file, compressed when asked and the browser can.
export async function offerDownload(text, name, { compress = false } = {}) {
  const squeeze = compress && typeof CompressionStream === 'function';
  const blob = squeeze ? await gzip(text) : new Blob([text], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = squeeze ? `${name}.gz` : name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  return { bytes: blob.size, name: link.download };
}
