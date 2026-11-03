// Branch tree for one assistant answer. Every branch stores exact token IDs:
// a branch copies its parent's path up to the fork position, commits the
// chosen alternative there, then appends what the model generated after it.
// The engine replays `inputTokenIds + path` exactly; display text is derived.

/** @param {unknown} value @param {string} label @returns {asserts value is number[]} */
function assertTokenIds(value, label) {
  if (!Array.isArray(value) || value.some((id) => !Number.isSafeInteger(id) || id < 0)) {
    throw new Error(`${label} must be an array of non-negative safe integer token IDs.`);
  }
}

/** @param {import('./branches.js').BranchTree} tree @param {number} id @returns {import('./branches.js').BranchNode} */
function nodeById(tree, id) {
  const node = tree.nodes.get(id);
  if (!node) throw new Error(`Unknown branch ${id}.`);
  return node;
}

/** @type {import('./branches.js').createBranchTree} */
export function createBranchTree({ inputTokenIds, tokenIds, records }) {
  assertTokenIds(inputTokenIds, 'Branch tree inputTokenIds');
  if (inputTokenIds.length === 0) throw new Error('Branch tree inputTokenIds must not be empty.');
  assertTokenIds(tokenIds, 'Branch tree tokenIds');
  if (!Array.isArray(records) || records.length !== tokenIds.length
    || records.some((record, index) => record?.tokenId !== tokenIds[index])) {
    throw new Error('Branch tree needs one inspection record per generated token, in order.');
  }
  /** @type {import('./branches.js').BranchNode} */
  const root = {
    id: 0,
    parentId: null,
    forkIndex: 0,
    pathIds: [...tokenIds],
    records: records.map((record, index) => ({ ...record, index })),
    status: 'complete',
  };
  return { inputTokenIds: [...inputTokenIds], nodes: new Map([[0, root]]), activeId: 0, nextId: 1 };
}

/** @type {import('./branches.js').activeBranch} */
export function activeBranch(tree) {
  return nodeById(tree, tree.activeId);
}

/** @type {import('./branches.js').branchChildren} */
export function branchChildren(tree, parentId, forkIndex = null) {
  return [...tree.nodes.values()].filter((node) => node.parentId === parentId
    && (forkIndex === null || node.forkIndex === forkIndex));
}

/** @type {import('./branches.js').selectBranch} */
export function selectBranch(tree, id) {
  const node = nodeById(tree, id);
  tree.activeId = id;
  return node;
}

// Starts a branch from the active path at `forkIndex` with `tokenId`. Picking
// the token already on the path, or one an existing sibling chose, selects
// that branch instead of generating a duplicate.
/** @type {import('./branches.js').startBranch} */
export function startBranch(tree, forkIndex, tokenId) {
  const parent = activeBranch(tree);
  if (!Number.isSafeInteger(forkIndex) || forkIndex < 0 || forkIndex >= parent.records.length) {
    throw new Error(`Branch position ${forkIndex} is outside the active answer.`);
  }
  if (parent.pathIds[forkIndex] === tokenId) return { node: parent, created: false };
  const origin = parent.records[forkIndex];
  const candidate = (origin.topCandidates ?? []).find((entry) => entry.tokenId === tokenId);
  if (!candidate) throw new Error(`Token ${tokenId} is not a recorded alternative at position ${forkIndex}.`);
  // Siblings share the parent's path before the fork, so match on the parent
  // whose path owns this position.
  const owner = forkOwner(tree, parent, forkIndex);
  const existing = branchChildren(tree, owner.id, forkIndex).find((node) => node.pathIds[forkIndex] === tokenId)
    ?? (owner.pathIds[forkIndex] === tokenId ? owner : null);
  if (existing) {
    tree.activeId = existing.id;
    return { node: existing, created: false };
  }
  const probability = Number.isFinite(candidate.probability) ? candidate.probability : null;
  /** @type {import('./branches.js').BranchNode} */
  const node = {
    id: tree.nextId++,
    parentId: owner.id,
    forkIndex,
    pathIds: [...parent.pathIds.slice(0, forkIndex), tokenId],
    records: [
      ...parent.records.slice(0, forkIndex),
      {
        index: forkIndex,
        tokenId,
        text: candidate.text,
        probability,
        surprisal: probability && probability > 0 ? -Math.log(probability) : null,
        // Same distribution as the replaced token, so alternatives stay available here.
        topCandidates: origin.topCandidates,
        chosen: true,
      },
    ],
    status: 'running',
  };
  tree.nodes.set(node.id, node);
  tree.activeId = node.id;
  return { node, created: true };
}

// The branch whose own tokens occupy `forkIndex` on the active path: the
// deepest ancestor that forked at or before that position.
/** @param {import('./branches.js').BranchTree} tree @param {import('./branches.js').BranchNode} node @param {number} forkIndex @returns {import('./branches.js').BranchNode} */
function forkOwner(tree, node, forkIndex) {
  let current = node;
  while (current.parentId !== null && current.forkIndex > forkIndex) {
    current = nodeById(tree, current.parentId);
  }
  if (current.parentId !== null && current.forkIndex === forkIndex) {
    return nodeById(tree, current.parentId);
  }
  return current;
}

/** @type {import('./branches.js').branchInputTokenIds} */
export function branchInputTokenIds(tree, node) {
  return [...tree.inputTokenIds, ...node.pathIds];
}

/** @type {import('./branches.js').appendBranchToken} */
export function appendBranchToken(node, tokenId, record = null) {
  const index = node.pathIds.length;
  node.pathIds.push(tokenId);
  node.records.push(record ? { ...record, index } : { index, tokenId, text: '', probability: null, topCandidates: [] });
}

// Replaces streamed records with the receipt's authoritative ones.
/** @type {import('./branches.js').settleBranch} */
export function settleBranch(node, status, generatedTokenIds = null, records = null) {
  if (generatedTokenIds && records) {
    const start = node.forkIndex + 1;
    node.pathIds = [...node.pathIds.slice(0, start), ...generatedTokenIds];
    node.records = [
      ...node.records.slice(0, start),
      ...records.map((record, offset) => ({ ...record, index: start + offset })),
    ];
  }
  node.status = status;
  return node;
}

/** @type {import('./branches.js').removeBranch} */
export function removeBranch(tree, node) {
  if (node.parentId === null) throw new Error('The original answer cannot be removed.');
  for (const child of branchChildren(tree, node.id)) removeBranch(tree, child);
  tree.nodes.delete(node.id);
  if (!tree.nodes.has(tree.activeId) && node.parentId !== null) tree.activeId = node.parentId;
}

// Fork switchers for the active path: position -> { members, active }.
// Members are branch IDs in creation order; `active` indexes the one shown.
/** @type {import('./branches.js').forkGroups} */
export function forkGroups(tree) {
  const chain = [];
  for (let node = activeBranch(tree); ; node = nodeById(tree, node.parentId)) {
    chain.unshift(node);
    if (node.parentId === null) break;
  }
  const groups = new Map();
  for (const [depth, node] of chain.entries()) {
    const next = chain[depth + 1] ?? null;
    const limit = next ? next.forkIndex : Infinity;
    const positions = new Set(branchChildren(tree, node.id)
      .map((child) => child.forkIndex)
      .filter((position) => position <= limit));
    for (const position of positions) {
      const members = [node.id, ...branchChildren(tree, node.id, position).map((child) => child.id)];
      const shown = next && next.forkIndex === position ? next.id : node.id;
      groups.set(position, { members, active: members.indexOf(shown) });
    }
  }
  return groups;
}

// Groups token fragments for display. A token whose text is an incomplete
// UTF-8 sequence joins the following tokens until they decode completely; the
// group still forks at its first token's exact boundary.
/** @type {import('./branches.js').groupTokenFragments} */
export function groupTokenFragments(records, decodeTokenIds = null) {
  const groups = [];
  for (let index = 0; index < records.length; index += 1) {
    const start = index;
    const ids = [records[index].tokenId];
    let text = String(records[index].text ?? '');
    while (text.includes('�') && index + 1 < records.length) {
      index += 1;
      ids.push(records[index].tokenId);
      text = typeof decodeTokenIds === 'function'
        ? String(decodeTokenIds(ids))
        : ids.map((_, offset) => String(records[start + offset].text ?? '')).join('');
    }
    groups.push({ start, end: index + 1, text });
  }
  return groups;
}

/** @type {import('./branches.js').isBranchPoint} */
export function isBranchPoint(record, threshold, showAll) {
  if (showAll) return true;
  const probability = record?.probability;
  return typeof probability === 'number' && Number.isFinite(probability) && probability < threshold;
}
