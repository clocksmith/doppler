export function compareTuple(a, b) {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  if (a.length < b.length) return -1;
  if (a.length > b.length) return 1;
  return 0;
}

export class MinHeap {
  constructor(compare) {
    this.compare = compare;
    this.data = [];
  }

  push(item) {
    const data = this.data;
    data.push(item);
    let idx = data.length - 1;
    while (idx > 0) {
      const parent = Math.floor((idx - 1) / 2);
      if (this.compare(data[idx], data[parent]) >= 0) break;
      [data[idx], data[parent]] = [data[parent], data[idx]];
      idx = parent;
    }
  }

  pop() {
    const data = this.data;
    if (!data.length) return null;
    const root = data[0];
    const tail = data.pop();
    if (data.length && tail) {
      data[0] = tail;
      let idx = 0;
      while (true) {
        const left = idx * 2 + 1;
        const right = left + 1;
        let smallest = idx;
        if (left < data.length && this.compare(data[left], data[smallest]) < 0) {
          smallest = left;
        }
        if (right < data.length && this.compare(data[right], data[smallest]) < 0) {
          smallest = right;
        }
        if (smallest === idx) break;
        [data[idx], data[smallest]] = [data[smallest], data[idx]];
        idx = smallest;
      }
    }
    return root;
  }

  peek() {
    return this.data.length ? this.data[0] : null;
  }

  get size() {
    return this.data.length;
  }
}
