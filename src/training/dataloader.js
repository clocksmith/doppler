function shuffledIndices(length) {
  const indices = Array.from({ length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
  return indices;
}

export class DataLoader {
  constructor(dataset, batchSize, shuffle = true) {
    this.dataset = dataset;
    this.batchSize = batchSize;
    this.shuffle = shuffle;
  }

  collate(batch) {
    return batch;
  }

  async *batches() {
    const length = this.dataset.length ?? 0;
    const indices = this.shuffle ? shuffledIndices(length) : Array.from({ length }, (_, i) => i);
    for (let i = 0; i < indices.length; i += this.batchSize) {
      const batchIndices = indices.slice(i, i + this.batchSize);
      const batch = batchIndices.map((idx) => this.dataset[idx]);
      yield this.collate(batch);
    }
  }
}
