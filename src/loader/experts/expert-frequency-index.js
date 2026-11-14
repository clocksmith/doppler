// Indexed resident heap; history is bounded by the model's declared expert keys.
export class ExpertFrequencyIndex {
  #limit;
  #sequence = 0;
  /** @type {Map<string, import('./expert-frequency-index.js').ExpertFrequencyEntry>} */
  #history = new Map();
  /** @type {import('./expert-frequency-index.js').ExpertFrequencyEntry[]} */
  #heap = [];
  /** @type {Map<string, number>} */
  #positions = new Map();

  /** @param {number} maxHistoryEntries */
  constructor(maxHistoryEntries) {
    if (!Number.isSafeInteger(maxHistoryEntries) || maxHistoryEntries < 1) {
      throw new Error('Expert frequency history requires a positive declared key limit.');
    }
    this.#limit = maxHistoryEntries;
  }

  get historySize() { return this.#history.size; }
  get residentSize() { return this.#heap.length; }

  /** @param {string} key */
  observe(key) {
    if (typeof key !== 'string' || key.length === 0) throw new Error('Expert frequency key must be nonempty.');
    let entry = this.#history.get(key);
    if (!entry && this.#history.size >= this.#limit) throw new Error('Expert frequency history exceeds declared key limit.');
    if (this.#sequence === Number.MAX_SAFE_INTEGER || entry?.frequency === Number.MAX_SAFE_INTEGER) {
      throw new Error('Expert frequency counter exceeds safe integer range.');
    }
    if (!entry) {
      entry = { key, frequency: 0, sequence: 0 };
      this.#history.set(key, entry);
    }
    entry.frequency++;
    entry.sequence = ++this.#sequence;
    const position = this.#positions.get(key);
    if (position !== undefined) this.#down(position);
    return entry.frequency;
  }

  /** @param {string} key */
  admit(key) {
    const entry = this.#history.get(key);
    if (!entry) throw new Error('Expert admission requires an observed key.');
    if (this.#positions.has(key)) return;
    this.#positions.set(key, this.#heap.length);
    this.#heap.push(entry);
    this.#up(this.#heap.length - 1);
  }

  /** @param {string} key */
  remove(key) {
    const position = this.#positions.get(key);
    if (position === undefined) return false;
    this.#removeAt(position);
    return true;
  }

  /** @param {(key: string) => boolean} isPinned */
  oldestUnpinned(isPinned) {
    const pinned = [];
    try {
      while (this.#heap.length) {
        const key = this.#heap[0].key;
        if (!isPinned(key)) return key;
        pinned.push(key);
        this.#removeAt(0);
      }
      return null;
    } finally {
      for (const key of pinned) this.admit(key);
    }
  }

  clear() {
    this.#history.clear();
    this.#positions.clear();
    this.#heap.length = 0;
    this.#sequence = 0;
  }

  /** @param {number} a @param {number} b */
  #less(a, b) {
    const x = this.#heap[a], y = this.#heap[b];
    return x.frequency < y.frequency || (x.frequency === y.frequency && x.sequence < y.sequence);
  }

  /** @param {number} a @param {number} b */
  #swap(a, b) {
    [this.#heap[a], this.#heap[b]] = [this.#heap[b], this.#heap[a]];
    this.#positions.set(this.#heap[a].key, a);
    this.#positions.set(this.#heap[b].key, b);
  }

  /** @param {number} index */
  #up(index) {
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (!this.#less(index, parent)) break;
      this.#swap(index, parent);
      index = parent;
    }
  }

  /** @param {number} index */
  #down(index) {
    while (index * 2 + 1 < this.#heap.length) {
      let child = index * 2 + 1;
      if (child + 1 < this.#heap.length && this.#less(child + 1, child)) child++;
      if (!this.#less(child, index)) break;
      this.#swap(index, child);
      index = child;
    }
  }

  /** @param {number} index */
  #removeAt(index) {
    const removed = this.#heap[index];
    const last = this.#heap.pop();
    this.#positions.delete(removed.key);
    if (index === this.#heap.length || !last) return;
    this.#heap[index] = last;
    this.#positions.set(last.key, index);
    const parent = Math.floor((index - 1) / 2);
    if (index > 0 && this.#less(index, parent)) this.#up(index);
    else this.#down(index);
  }
}
