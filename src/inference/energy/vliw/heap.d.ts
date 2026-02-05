export function compareTuple(a: number[], b: number[]): number;
export class MinHeap<T = unknown> {
  constructor(compare: (a: T, b: T) => number);
  push(item: T): void;
  pop(): T | null;
  peek(): T | null;
  readonly size: number;
}
