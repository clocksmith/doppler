export interface ExpertFrequencyEntry { key: string; frequency: number; sequence: number }
/** Bounded lifetime-frequency history and indexed resident priority heap. */
export declare class ExpertFrequencyIndex {
  constructor(maxHistoryEntries: number);
  readonly historySize: number;
  readonly residentSize: number;
  observe(key: string): number;
  admit(key: string): void;
  remove(key: string): boolean;
  /** Returns the least frequent, then least recently observed resident without removing it. */
  oldestUnpinned(isPinned: (key: string) => boolean): string | null;
  clear(): void;
}
