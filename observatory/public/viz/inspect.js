// The one place a drawn value is shown exactly: what it is, where it came
// from, at what precision and on which device. Every view opens this, so every
// click on a mark behaves the same.

import { describeProvenance, provenanceMark } from './provenance.js';

// What the inspector shows for an entry. What a run did not record is said so.
export function inspectorRows({ title, value, provenance, source = null, precision = null, device = null, rows = [] }) {
  if (!title) throw new Error('An inspected value needs a title.');
  describeProvenance(provenance);
  if (value === undefined) throw new Error('An inspected value needs a value; pass null for one that was not recorded.');
  const said = (text) => (text === null || text === undefined || text === '' ? 'not recorded' : String(text));
  return [['Value', said(value)], ...rows.map(([term, text]) => [term, said(text)]), ['Source', said(source)], ['Precision', said(precision)], ['Device', said(device)]];
}

let panel = null;
let opener = null;

export function closeInspector() {
  if (!panel) return;
  panel.hidden = true;
  if (opener?.isConnected) opener.focus();
  opener = null;
}

export function openInspector(entry, anchor = null) {
  const rows = inspectorRows(entry);
  if (!panel) {
    panel = document.createElement('aside');
    panel.id = 'site-inspector';
    panel.className = 'panel site-inspector';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', 'Value details');
    panel.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeInspector(); });
    (document.getElementById('app') ?? document.body).append(panel);
  }
  const title = document.createElement('strong');
  title.textContent = entry.title;
  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'btn btn-small';
  close.textContent = 'Close';
  close.addEventListener('click', closeInspector);
  const head = document.createElement('div');
  head.className = 'panel-header';
  head.append(title, provenanceMark(entry.provenance), close);
  const list = document.createElement('dl');
  list.className = 'site-facts';
  for (const [term, text] of rows) {
    const dt = document.createElement('dt'); dt.textContent = term;
    const dd = document.createElement('dd'); dd.textContent = text;
    list.append(dt, dd);
  }
  const note = document.createElement('p');
  note.className = 'demo-note';
  note.textContent = describeProvenance(entry.provenance).what;
  const body = document.createElement('div');
  body.className = 'panel-body';
  body.append(list, note);
  panel.replaceChildren(head, body);
  panel.hidden = false;
  opener = anchor;
  close.focus();
}
