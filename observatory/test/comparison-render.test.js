import test from 'node:test';
import assert from 'node:assert/strict';
import { renderForest } from '../public/views/forest.js';
import { renderConcertina } from '../public/views/concertina.js';
import { renderFaultline } from '../public/views/faultline.js';
import { createSelection } from '../public/selection.js';

class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.attrs = {}; this.events = {}; this.style = { setProperty() {} }; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  addEventListener(key, handler) { this.events[key] = handler; }
  focus() { this.focused = true; }
}
function descendants(node, tag) { return node.children.filter((child) => child instanceof Node).flatMap((child) => [...(child.tag === tag ? [child] : []), ...descendants(child, tag)]); }
function text(node) { return typeof node === 'string' ? node : [node.textContent ?? '', ...node.children.map(text)].join(' '); }
function setup(t) {
  const roots = {};
  globalThis.document = {
    createElement: (tag) => new Node(tag),
    getElementById(id) {
      if (!roots[id]) { roots[id] = new Node('div'); roots[id].tBodies = [new Node('tbody')]; }
      return roots[id];
    },
  };
  t.after(() => { delete globalThis.document; });
  document.getElementById('tolerance').value = '0.001';
  return document.getElementById;
}
function run(id = 'run-a') {
  return { id, createdAt: '2026-11-12', parent: null, inputTokenIds: [1], prompt: 'p', outputText: 'a b',
    model: { manifestHash: 'same', tokenizerDigest: 'same' }, environment: {}, observation: {}, sampling: { maxTokens: 3 },
    tokens: [7, 8].map((tokenId, index) => ({ index, tokenId, text: String(tokenId), probability: 0.6, candidates: [{ tokenId, text: String(tokenId), probability: 0.6 }, { tokenId: 9, text: '9', probability: 0.3 }] })),
    timing: { gpu: null }, layerOutputs: null };
}

test('Branch here passes the recorded token ID; nodes update shared run and branch selection', (t) => {
  const $ = setup(t); const a = run(); const b = run('run-b'); b.parent = { runId: a.id, index: 0, tokenId: 9, tokenText: '9' };
  // Answer token 2 (position 2): its alternative 9 has not been run.
  const selection = createSelection({ runId: a.id, position: 2 });
  let request;
  renderForest({ run: a, runs: [a, b], selection, canRun: true, actions: { branch: (...args) => { request = args; }, showRun() {} } });
  descendants($('candidates').tBodies[0], 'button').find((node) => text(node).trim() === 'Branch here').events.click();
  assert.deepEqual(request, [a, 1, 9]);
  descendants($('forest'), 'button')[1].events.click();
  assert.equal(selection.get().runId, b.id); assert.equal(selection.get().branchId, b.id);
});

test('forest renders cyclic parent links once and missing alternatives without throwing', (t) => {
  const $ = setup(t); const a = run(); const b = run('run-b');
  a.parent = { runId: b.id, index: 0, tokenText: '7' }; b.parent = { runId: a.id, index: 0, tokenText: '8' };
  a.tokens[0].candidates = null;
  renderForest({ run: a, runs: [a, b], selection: createSelection({ position: 1 }), canRun: false, actions: {} });
  assert.equal(descendants($('forest'), 'button').length, 2);
  assert.match($('forest-note').textContent, /cycle/);
  assert.match($('token-note').textContent, /did not record alternatives/);
});

test('concertina handles absent and empty timing, zero durations and operation selection', (t) => {
  const $ = setup(t); const a = run(); const selection = createSelection({ position: 2 });
  const context = { run: a, selection, canRun: false, actions: {} };
  renderConcertina(context); assert.equal($('timing').hidden, true);
  a.timing.gpu = { steps: [] }; renderConcertina(context); assert.equal($('timing').hidden, true);
  a.timing.gpu = { steps: [{ step: 1, totalMs: 0, kernels: { matmul: 0 } }], prefillMs: null };
  renderConcertina(context); assert.equal($('timing').hidden, false);
  assert.doesNotMatch(text($('step-detail')), /NaN|Infinity/);
  descendants($('step-detail'), 'button')[0].events.click(); assert.equal(selection.get().op, 'matmul');
  descendants($('time-strip'), 'button')[0].events.click(); assert.equal(selection.get().position, null);
});

test('faultline handles absent comparison, missing layers, and keyboard selection of recorded cells', (t) => {
  const $ = setup(t); const a = run(); const b = run('run-b'); const selection = createSelection();
  renderFaultline({ run: a, other: null, selection }); assert.equal($('compare-view').hidden, true);
  renderFaultline({ run: a, other: b, selection }); assert.match($('compare-layers-note').textContent, /did not record layer outputs/);
  a.layerOutputs = { positions: [0], layers: 1, width: 1, missing: [], decoded: new Float32Array([1]) };
  b.layerOutputs = { ...a.layerOutputs, decoded: new Float32Array([2]) };
  renderFaultline({ run: a, other: b, selection });
  let prevented = false;
  const cell = descendants($('heat').tBodies[0], 'td')[0];
  cell.events.keydown({ key: 'Enter', preventDefault() { prevented = true; } });
  assert.equal(prevented, true); assert.equal(selection.get().position, 0); assert.equal(selection.get().layer, 0);
  assert.match(text($('compare-layers-note')), /Coverage: 1\/1/);
});
