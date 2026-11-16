// Most language pretokens are short. Bound the scan work there to avoid heap
// and linked-list allocations; long identifiers take the indexed path below.
const SCAN_MAX_CHARACTERS = 32;

/** @param {string[]} tokens @param {Pick<ReadonlyMap<string, number>, 'get'>} ranks */
function mergeShortPretoken(tokens, ranks) {
  while (tokens.length > 1) {
    let best = Infinity;
    let pair = null;
    for (let index = 0; index + 1 < tokens.length; index++) {
      const key = `${tokens[index]} ${tokens[index + 1]}`;
      const rank = ranks.get(key);
      if (rank !== undefined && rank < best) {
        best = rank;
        pair = key;
      }
    }
    if (pair === null) break;
    const [left, right] = pair.split(' ');
    const merged = [];
    for (let index = 0; index < tokens.length; index++) {
      if (tokens[index] === left && tokens[index + 1] === right) {
        merged.push(left + right);
        index++;
      } else merged.push(tokens[index]);
    }
    tokens = merged;
  }
  return tokens;
}

// Keep only adjacent candidates in the rank heap. A merge updates its two
// neighbors; it never rescans or copies the rest of the pretoken.
/** @type {typeof import('./merge.js').mergeBpeTokens} */
export function mergeBpeTokens(text, ranks) {
  const tokens = text.split('');
  const count = tokens.length;
  if (count < 2) return tokens;
  if (count <= SCAN_MAX_CHARACTERS) return mergeShortPretoken(tokens, ranks);
  const previous = new Int32Array(count);
  const next = new Int32Array(count);
  const versions = new Uint32Array(count);
  /** @type {Array<{ index: number, version: number, rank: number }>} */
  const heap = [];

  /** @param {{ index: number, version: number, rank: number }} entry */
  function push(entry) {
    let index = heap.length;
    heap.push(entry);
    while (index > 0) {
      const parent = (index - 1) >>> 1;
      if (heap[parent].rank <= entry.rank) break;
      heap[index] = heap[parent];
      index = parent;
    }
    heap[index] = entry;
  }

  function pop() {
    const first = heap[0];
    const tail = heap.pop();
    if (heap.length > 0 && tail) {
      let index = 0;
      while (index * 2 + 1 < heap.length) {
        let child = index * 2 + 1;
        if (child + 1 < heap.length && heap[child + 1].rank < heap[child].rank) child++;
        if (tail.rank <= heap[child].rank) break;
        heap[index] = heap[child];
        index = child;
      }
      heap[index] = tail;
    }
    return first;
  }

  /** @param {number} index */
  function addCandidate(index) {
    if (index < 0 || next[index] < 0) return;
    const rank = ranks.get(`${tokens[index]} ${tokens[next[index]]}`);
    if (rank !== undefined && rank < Infinity) push({ index, version: versions[index], rank });
  }

  /** @param {{ index: number, version: number, rank: number }} entry */
  function valid(entry) {
    return next[entry.index] >= 0 && versions[entry.index] === entry.version;
  }

  for (let index = 0; index < count; index++) {
    previous[index] = index - 1;
    next[index] = index + 1 < count ? index + 1 : -1;
  }
  for (let index = 0; index < count - 1; index++) addCandidate(index);

  while (heap.length > 0) {
    const first = pop();
    if (!valid(first)) continue;
    const candidates = [first];
    // The existing tokenizer merges every nonoverlapping occurrence of the
    // selected pair in one round. Freeze this round before adding new pairs:
    // newly created lower ranks must not interrupt its remaining occurrences.
    while (heap.length > 0 && heap[0].rank === first.rank) {
      const entry = pop();
      if (valid(entry)) candidates.push(entry);
    }
    candidates.sort((a, b) => a.index - b.index);
    const changed = new Set();
    for (const entry of candidates) {
      if (!valid(entry)) continue;
      const left = entry.index;
      const right = next[left];
      tokens[left] += tokens[right];
      next[left] = next[right];
      versions[left]++;
      next[right] = -2;
      versions[right]++;
      if (next[left] >= 0) previous[next[left]] = left;
      const before = previous[left];
      if (before >= 0) {
        versions[before]++;
        changed.add(before);
      }
      changed.add(left);
    }
    for (const index of changed) addCandidate(index);
  }

  const result = [];
  for (let index = 0; index >= 0; index = next[index]) result.push(tokens[index]);
  return result;
}
