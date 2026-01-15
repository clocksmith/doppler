export declare class DataLoader<T> {
  constructor(dataset: T[], batchSize: number, shuffle?: boolean);
  dataset: T[];
  batchSize: number;
  shuffle: boolean;
  collate(batch: T[]): T[] | unknown;
  batches(): AsyncGenerator<T[] | unknown, void, void>;
}
