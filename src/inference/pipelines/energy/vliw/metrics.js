export function computeTopologicalOrder(graph) {
  const indeg = graph.indeg.slice();
  const order = [];
  const queue = [];
  for (let i = 0; i < indeg.length; i++) {
    if (indeg[i] === 0) queue.push(i);
  }
  while (queue.length) {
    const node = queue.shift();
    order.push(node);
    const next = graph.succ[node];
    for (let i = 0; i < next.length; i++) {
      const succ = next[i];
      indeg[succ] -= 1;
      if (indeg[succ] === 0) queue.push(succ);
    }
  }
  return order;
}

export function computeGraphMetrics(graph) {
  const n = graph.indeg.length;
  const order = computeTopologicalOrder(graph);
  const height = new Float32Array(n);
  const earliest = new Int32Array(n);
  const latest = new Int32Array(n);

  if (order.length !== n) {
    return {
      height,
      slack: new Float32Array(n),
      order,
    };
  }

  for (let i = order.length - 1; i >= 0; i--) {
    const node = order[i];
    const next = graph.succ[node];
    let maxChild = 0;
    for (let j = 0; j < next.length; j++) {
      const succ = next[j];
      if (height[succ] > maxChild) maxChild = height[succ];
    }
    height[node] = maxChild + 1;
  }

  for (let i = 0; i < order.length; i++) {
    const node = order[i];
    const next = graph.succ[node];
    for (let j = 0; j < next.length; j++) {
      const succ = next[j];
      const candidate = earliest[node] + 1;
      if (candidate > earliest[succ]) earliest[succ] = candidate;
    }
  }

  let maxPath = 0;
  for (let i = 0; i < n; i++) {
    if (earliest[i] > maxPath) maxPath = earliest[i];
  }
  maxPath += 1;

  latest.fill(maxPath - 1);
  for (let i = order.length - 1; i >= 0; i--) {
    const node = order[i];
    const next = graph.succ[node];
    if (!next.length) continue;
    let minLatest = Number.POSITIVE_INFINITY;
    for (let j = 0; j < next.length; j++) {
      const succ = next[j];
      const candidate = latest[succ] - 1;
      if (candidate < minLatest) minLatest = candidate;
    }
    if (Number.isFinite(minLatest)) {
      latest[node] = Math.min(latest[node], minLatest);
    }
  }

  const slack = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const value = latest[i] - earliest[i];
    slack[i] = value >= 0 ? value : 0;
  }

  return {
    height,
    slack,
    order,
  };
}
