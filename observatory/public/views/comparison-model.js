// Pure view models. Missing observations stay missing; topology is not evidence
// of probabilities for a complete continuation.
export function forestRows(selected, runs) {
  const byId = new Map(runs.map((run) => [run.id, run]));
  const ancestry = new Set();
  let root = selected;
  while (root.parent && byId.has(root.parent.runId) && !ancestry.has(root.parent.runId)) {
    ancestry.add(root.id);
    root = byId.get(root.parent.runId);
  }
  const rows = [];
  const visited = new Set();
  const pending = [{ run: root, depth: 0 }];
  while (pending.length) {
    const row = pending.pop();
    if (visited.has(row.run.id)) continue;
    visited.add(row.run.id);
    rows.push(row);
    const children = runs.filter((run) => run.parent?.runId === row.run.id)
      .sort((a, b) => a.parent.index - b.parent.index || a.createdAt.localeCompare(b.createdAt));
    for (let i = children.length - 1; i >= 0; i--) pending.push({ run: children[i], depth: row.depth + 1 });
  }
  return { root, rows, cyclic: Boolean(root.parent && ancestry.has(root.parent.runId)) };
}

export function operationBreakdown(step) {
  const operations = Object.entries(step.kernels)
    .filter(([, time]) => Number.isFinite(time) && time >= 0)
    .sort((a, b) => b[1] - a[1])
    .map(([name, time]) => ({ name, time, share: step.totalMs > 0 ? time / step.totalMs : null }));
  const sum = operations.reduce((total, operation) => total + operation.time, 0);
  return { operations, sum, remainder: Math.max(0, step.totalMs - sum), exceedsTotal: sum > step.totalMs + Math.max(1e-6, step.totalMs * 1e-6) };
}

export function comparableCoverage(layers) {
  if (!layers) return null;
  const measured = layers.relative.flat().filter(Number.isFinite).length;
  const expected = layers.positions.length * layers.layerCount;
  return { measured, expected, complete: expected > 0 && measured === expected };
}
