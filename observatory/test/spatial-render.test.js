import test from 'node:test';
import assert from 'node:assert/strict';
import { renderAtlas } from '../public/views/atlas.js';
import { renderJourney } from '../public/views/journey.js';

class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.attrs = {}; this.events = {}; this.style = { setProperty() {} }; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  addEventListener(key, handler) { this.events[key] = handler; }
}
function descendants(node, tag) { return node.children.filter((child) => child instanceof Node).flatMap((child) => [...(child.tag === tag ? [child] : []), ...descendants(child, tag)]); }
function documentFor(ids) {
  const roots = Object.fromEntries(ids.map((id) => [id, new Node('div')]));
  return { roots, createElement: (tag) => new Node(tag), createElementNS: (_, tag) => new Node(tag), getElementById: (id) => roots[id] };
}

test('atlas window cell selects original answer index, not window offset', () => {
  globalThis.document = documentFor(['atlas']);
  const run = { inputTokenIds: [1, 2], tokens: Array.from({ length: 100 }, (_, i) => ({ tokenId: 7, text: 'a', candidates: [{ tokenId: 7, text: 'a', probability: 0.5 }] })) };
  let chosen;
  renderAtlas({ run, selection: { get: () => ({ position: 82 }), select: (value) => { chosen = value; } } });
  const buttons = descendants(document.roots.atlas, 'button');
  assert.equal(buttons.length, 64);
  buttons[0].events.click();
  assert.deepEqual(chosen, { position: 38 });
  delete globalThis.document;
});

test('journey renders finite captured points and does not connect across a missing layer', () => {
  globalThis.document = documentFor(['journey']);
  const run = { tokens: [], inputTokenIds: [1], layerOutputs: { layers: 3, width: 2, positions: [0], missing: [[0, 1]], decoded: new Float32Array([0, 1, 0, 0, 2, 3]) }, observation: {}, environment: {} };
  renderJourney({ run, selection: { get: () => ({ position: 0, layer: 0 }), select() {} } });
  assert.equal(descendants(document.roots.journey, 'circle').length, 2);
  assert.equal(descendants(document.roots.journey, 'line').length, 2, 'only the two axes, no interpolated path');
  const mode = descendants(document.roots.journey, 'select')[0];
  mode.events.change({ target: { value: '2' } });
  assert.equal(descendants(document.roots.journey, 'input')[0].disabled, true);
  delete globalThis.document;
});
