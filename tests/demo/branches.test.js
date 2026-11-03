import assert from 'node:assert/strict';

import {
  activeBranch,
  appendBranchToken,
  branchInputTokenIds,
  createBranchTree,
  forkGroups,
  groupTokenFragments,
  isBranchPoint,
  removeBranch,
  selectBranch,
  settleBranch,
  startBranch,
} from '../../demo/branches.js';

const candidates = (selected, ...others) => [selected, ...others].map((tokenId, rank) => ({
  tokenId,
  text: `<${tokenId}>`,
  probability: rank === 0 ? 0.6 : 0.4 / others.length,
}));
const record = (tokenId, text, probability, alternatives = [tokenId]) => ({
  tokenId,
  text,
  probability,
  topCandidates: candidates(...alternatives),
});

// Original answer: tokens 10..14 after prompt [1, 2, 3].
const tree = createBranchTree({
  inputTokenIds: [1, 2, 3],
  tokenIds: [10, 11, 12, 13, 14],
  records: [
    record(10, 'The', 0.99),
    record(11, ' sky', 0.42, [11, 21, 31]),
    record(12, ' is', 0.95),
    record(13, ' blue', 0.3, [13, 23, 33]),
    record(14, '.', 0.9),
  ],
});
const root = activeBranch(tree);
assert.equal(root.parentId, null);
assert.deepEqual(forkGroups(tree), new Map(), 'An answer without branches has no switchers');

// Branching replays the exact prompt IDs, the prefix IDs and the chosen ID.
const first = startBranch(tree, 3, 23);
assert.equal(first.created, true);
assert.deepEqual(first.node.pathIds, [10, 11, 12, 23]);
assert.deepEqual(branchInputTokenIds(tree, first.node), [1, 2, 3, 10, 11, 12, 23]);
assert.equal(first.node.records[3].chosen, true);
assert.deepEqual(first.node.records[3].topCandidates, root.records[3].topCandidates,
  'The chosen token keeps the distribution it was picked from');
assert.equal(first.node.status, 'running');
appendBranchToken(first.node, 40, record(40, ' green', 0.2, [40, 50]));
appendBranchToken(first.node, 41, record(41, '.', 0.97));
settleBranch(first.node, 'complete');
assert.deepEqual(first.node.pathIds, [10, 11, 12, 23, 40, 41]);
assert.deepEqual(first.node.records.map((entry) => entry.index), [0, 1, 2, 3, 4, 5]);
assert.deepEqual(forkGroups(tree), new Map([[3, { members: [0, first.node.id], active: 1 }]]));

// The receipt's records replace streamed ones after completion.
settleBranch(first.node, 'complete', [40, 42], [record(40, ' green', 0.2, [40, 50]), record(42, '!', 0.5)]);
assert.deepEqual(first.node.pathIds, [10, 11, 12, 23, 40, 42]);

// Picking the token already shown, or an alternative a sibling chose, reuses it.
assert.equal(startBranch(tree, 3, 23).created, false);
assert.equal(startBranch(tree, 0, 10).created, false, 'The shown token is not a new branch');
assert.throws(() => startBranch(tree, 3, 99), /not a recorded alternative/);
assert.throws(() => startBranch(tree, 9, 10), /outside the active answer/);

// Nested: branch the branch after its fork, then branch it before its fork.
const nested = startBranch(tree, 4, 50).node;
assert.equal(nested.parentId, first.node.id);
assert.deepEqual(branchInputTokenIds(tree, nested), [1, 2, 3, 10, 11, 12, 23, 50]);
settleBranch(nested, 'complete');
assert.deepEqual(forkGroups(tree), new Map([
  [3, { members: [0, first.node.id], active: 1 }],
  [4, { members: [first.node.id, nested.id], active: 1 }],
]));
// A fork before the nested branch's own fork belongs to the original answer.
const early = startBranch(tree, 1, 21).node;
assert.equal(early.parentId, root.id);
assert.deepEqual(early.pathIds, [10, 21]);
settleBranch(early, 'stopped');
assert.equal(early.status, 'stopped', 'A cancelled branch keeps its partial answer');
assert.deepEqual(forkGroups(tree), new Map([[1, { members: [0, early.id], active: 1 }]]));

// Switching back shows the original with every fork it owns.
selectBranch(tree, root.id);
assert.deepEqual(activeBranch(tree).pathIds, [10, 11, 12, 13, 14]);
assert.deepEqual(forkGroups(tree), new Map([
  [1, { members: [0, early.id], active: 0 }],
  [3, { members: [0, first.node.id], active: 0 }],
]));
selectBranch(tree, nested.id);
assert.deepEqual(activeBranch(tree).pathIds, [10, 11, 12, 23, 50]);

// A failed branch is removed with its descendants; the parent becomes active.
assert.equal(startBranch(tree, 4, 40).created, false);
assert.equal(activeBranch(tree).id, first.node.id, 'Picking the parent token at a fork returns to the parent branch');
const failing = startBranch(tree, 3, 33).node;
removeBranch(tree, failing);
assert.equal(tree.nodes.has(failing.id), false);
assert.equal(activeBranch(tree).id, failing.parentId);
assert.throws(() => removeBranch(tree, root), /cannot be removed/);

// Fragments of one character display together and fork at the first token.
const fragments = [
  { tokenId: 1, text: 'Caf' },
  { tokenId: 2, text: '�' },
  { tokenId: 3, text: '�' },
  { tokenId: 4, text: ' ok' },
  { tokenId: 5, text: '\n' },
  { tokenId: 6, text: '' },
];
const decode = (ids) => (ids.join(',') === '2,3' ? 'é' : ids.join(',') === '2' ? '�' : '?');
assert.deepEqual(groupTokenFragments(fragments, decode), [
  { start: 0, end: 1, text: 'Caf' },
  { start: 1, end: 3, text: 'é' },
  { start: 3, end: 4, text: ' ok' },
  { start: 4, end: 5, text: '\n' },
  { start: 5, end: 6, text: '' },
]);
assert.deepEqual(groupTokenFragments([{ tokenId: 9, text: '�' }], decode), [{ start: 0, end: 1, text: '�' }],
  'A trailing incomplete fragment stays visible');

// Threshold compares the selected token's probability; "all tokens" offers every token.
assert.equal(isBranchPoint({ probability: 0.49 }, 0.5, false), true);
assert.equal(isBranchPoint({ probability: 0.5 }, 0.5, false), false);
assert.equal(isBranchPoint({ probability: null }, 0.5, false), false);
assert.equal(isBranchPoint({ probability: 0.99 }, 0.5, true), true);

assert.throws(() => createBranchTree({ inputTokenIds: [], tokenIds: [1], records: [record(1, 'a', 1)] }), /must not be empty/);
assert.throws(() => createBranchTree({ inputTokenIds: [1], tokenIds: [1, 2], records: [record(1, 'a', 1)] }), /one inspection record/);

console.log('demo branches: ok');
