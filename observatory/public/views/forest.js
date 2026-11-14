// Continuation Forest: where the answer could have gone. At the selected token
// it shows the listed alternatives as a probability ribbon, with the unlisted
// probability as its own band, and below it the tree of branches that were
// actually run. An alternative that was not run stays "not run" until it is.

import { alternatives, answerIndex, forkRequest } from '../shared-run.js';
import { $, h, percent, visible } from '../viz/dom.js';
import { provenanceMark } from '../viz/provenance.js';
import { forestRows } from './comparison-model.js';
import { ribbon } from '../viz/ribbon.js';

export function renderForest({ run, runs, selection, why, canRun, actions }) {
  const index = answerIndex(run, selection.get().position);
  const view = index === null ? null : alternatives(run, index, runs);
  $('token-panel').hidden = index === null;
  if (index !== null) {
    const token = run.tokens[index];
    $('token-title').replaceChildren(`At answer token ${index + 1}: "${visible(token.text)}" `, view ? provenanceMark('measured', 'Next-token probabilities the model gave at this point.') : provenanceMark('missing'));
    const body = $('candidates').tBodies[0];
    if (!view) {
      body.replaceChildren();
      $('fork-ribbon').replaceChildren();
      $('token-note').textContent = 'This run did not record alternatives for its tokens.';
    } else {
      const unavailable = (candidate) => {
        if (!canRun) return why || 'A run is in progress.';
        try { forkRequest(run, index, candidate.tokenId); return ''; } catch (error) { return error.message; }
      };
      const pick = (candidate) => {
        if (candidate.state === 'run') actions.showRun(candidate.branchRunId);
        else if (candidate.state === 'not-run' && !unavailable(candidate)) actions.branch(run, index, candidate.tokenId);
      };
      $('fork-ribbon').replaceChildren(ribbon(view.candidates.map((candidate) => ({ ...candidate, key: candidate.tokenId, text: visible(candidate.text) })),
        { label: `What the model could have said at answer token ${index + 1}`, onPick: pick, unavailable: canRun ? '' : (why || 'A run is in progress.') }));
      body.replaceChildren(...view.candidates.map((candidate) => h('tr', {},
        h('td', {}, h('code', {}, visible(candidate.text))),
        h('td', { className: 'site-num' }, percent(candidate.probability)),
        h('td', {}, candidate.state === 'taken' ? 'taken by this run'
          : candidate.state === 'run' ? h('button', { className: 'btn btn-small', type: 'button', on: { click: () => pick(candidate) } }, 'Show this branch')
            : [provenanceMark('unrun'), canRun ? [' ', h('button', { className: 'btn btn-small', type: 'button', disabled: Boolean(unavailable(candidate)), title: unavailable(candidate), on: { click: () => pick(candidate) } }, 'Continue from here')] : null]),
      )), h('tr', {},
        h('td', { className: 'demo-note' }, `every other token (${view.candidates.length} are listed)`),
        h('td', { className: 'site-num demo-note' }, percent(view.unlistedProbability)),
        h('td', { className: 'demo-note' }, 'not listed, so not available to continue from'),
      ));
      const choices = view.candidates.filter((candidate) => candidate.tokenId !== token.tokenId);
      // A fork needs a model to run; without one the chooser would only offer a dead button.
      if (choices.length && canRun) {
        const chooser = h('select', { ariaLabel: 'Human fork: choose a recorded alternative token' },
          ...choices.map((candidate) => h('option', { value: String(candidate.tokenId) }, `${visible(candidate.text)} · ${percent(candidate.probability)}`)));
        const status = h('span', { className: 'demo-note', attrs: { role: 'status' } });
        const submit = h('button', { className: 'btn btn-small', type: 'button', disabled: !canRun,
          on: { click: () => {
            const candidate = choices.find((item) => item.tokenId === Number(chooser.value));
            const reason = unavailable(candidate);
            if (reason) { status.textContent = reason; return; }
            actions.branch(run, index, candidate.tokenId);
          } } }, 'Run my fork');
        $('fork-ribbon').append(h('div', { className: 'demo-controls' }, h('label', {}, 'Human fork ', chooser), submit, status),
          h('p', { className: 'demo-note' }, 'Continue from a recorded alternative.'));
      }
      const notRun = view.candidates.filter((candidate) => candidate.state === 'not-run').length;
      $('token-note').textContent = [notRun ? `${notRun} listed alternative${notRun === 1 ? ' has' : 's have'} not been run: what would follow is unknown until it is.` : '', why].filter(Boolean).join(' ');
    }
  }

  // Every opened descendant is selectable, including repeated forks from the
  // same token. Traverse defensively: imported parent links may form cycles.
  const { root, rows, cyclic } = forestRows(run, runs);
  $('forest').replaceChildren(...rows.map(({ run: node, depth }) => h('li', {
    style: `margin-inline-start: ${Math.min(depth, 4)}em`,
    data: { run: node.id, depth },
  },
    h('button', { type: 'button', className: 'btn btn-small', ariaPressed: String(node.id === run.id),
      on: { click: () => selection.select({ runId: node.id, branchId: node.parent ? node.id : null }) } },
    `${node.id === run.id ? 'Shown' : 'Show'} ${node.id.slice(4, 10)}`),
    node.parent ? h('span', { className: 'demo-note' }, ` Fork at token ${node.parent.index + 1}: replacement "${visible(node.parent.tokenText)}"${Number.isFinite(node.parent.probability) ? ` (${percent(node.parent.probability)} at the fork)` : ' (probability not recorded)'}. `) : ' Root. ',
    h('span', {}, node.outputText || '(no continuation text)'),
  )));
  $('forest-note').textContent = cyclic
    ? 'The imported branch links contain a cycle. Each available run is shown once; the lineage cannot be trusted.'
    : `${rows.length} run${rows.length === 1 ? '' : 's'}. ` + (root.parent
      ? 'The parent of this branch is not open. '
      : '') + 'A fork\'s probability is for its next token only, not the whole continuation.';
}
