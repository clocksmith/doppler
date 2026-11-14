import { answerIndex, sequencePosition } from '../shared-run.js';
import { $, h, percent, visible } from '../viz/dom.js';
import { gap } from '../viz/gap.js';
import { provenanceMark } from '../viz/provenance.js';
import { candidateAtlas } from './spatial-data.js';

export function renderAtlas({ run, selection }) {
  const root = $('atlas');
  if (!root) return;
  const selected = answerIndex(run, selection.get().position);
  const atlas = candidateAtlas(run, 24, { center: selected ?? 0 });
  if (!atlas.rows.length) {
    root.replaceChildren(gap({ what: 'No next-token candidate probabilities were recorded in this decision window. Select another answer token to move the window. This run format carries no vocabulary embeddings, so there is no semantic geometry to draw.' }));
    return;
  }
  root.replaceChildren(
    h('p', {}, provenanceMark('measured'), ' Candidate probability atlas. Rows are token IDs; columns are answer decisions. The same token can be likely at one decision and unlikely at another.'),
    h('p', { className: 'demo-note' }, 'This is a probability map, not a semantic embedding map. Darker cells mean higher recorded probability. A dash means unlisted, not zero. Rows are ordered by their highest recorded probability. Click a cell to select that answer decision in all views.'),
    h('div', { className: 'site-scroll', attrs: { tabindex: '0', 'aria-label': 'Candidate probability atlas' } }, h('table', { id: 'atlas-table' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Candidate'), ...run.tokens.slice(atlas.start, atlas.start + atlas.count).map((_, i) => h('th', {}, String(atlas.start + i + 1))))),
      h('tbody', {}, ...atlas.rows.map((row) => h('tr', {},
        h('th', { scope: 'row' }, visible(row.text ?? null), h('small', {}, ` [${row.tokenId}]`)),
        ...row.probabilities.map((probability, i) => h('td', {}, probability === null ? h('span', { title: 'Not listed at this decision' }, '—') : h('button', {
          className: `btn btn-small${selected === atlas.start + i ? ' is-selected' : ''}`,
          type: 'button', ariaPressed: String(selected === atlas.start + i),
          title: `Answer decision ${atlas.start + i + 1}: token ${row.tokenId}, ${percent(probability)}${run.tokens[atlas.start + i].tokenId === row.tokenId ? ', chosen' : ''}`,
          style: `background: color-mix(in srgb, currentColor ${Math.round(probability * 22)}%, transparent)`,
          on: { click: () => selection.select({ position: sequencePosition(run, atlas.start + i) }) },
        }, `${percent(probability)}${run.tokens[atlas.start + i].tokenId === row.tokenId ? ' ✓' : ''}`))),
      ))))),
    h('p', { className: 'demo-note' }, `Decisions ${atlas.start + 1}–${atlas.start + atlas.count} of ${run.tokens.length}; select another answer token to move this window. ${atlas.rows.length} of ${atlas.total} distinct recorded candidates shown${atlas.omitted ? `; ${atlas.omitted} lower-peak candidates omitted` : ''}. ✓ marks the chosen token. No vocabulary embedding distances or clusters are inferred.`),
  );
}
