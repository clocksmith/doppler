export interface RoutedExpertTraceOptions {
  requestId: string;
  maxRecords: number;
  maxIndices: number;
}
export interface RoutedExpertSelection {
  layer: number;
  phase: 'prefill' | 'decode' | 'unspecified';
  tokenOffset: number;
  numTokens: number;
  topK: number;
  source: 'paging' | 'resident';
}
export interface RoutedExpertTrace extends RoutedExpertTraceOptions {
  schema: 'doppler.routed-expert-trace/v1';
  scope: string;
  records: (RoutedExpertSelection & { indices: number[] })[];
  capturedIndices: number;
  omittedRecords: number;
  omittedIndices: number;
}
/** Caller brackets one idle-boundary request, including failures, with begin/finish. */
export function beginRoutedExpertTrace(loader: object, options: RoutedExpertTraceOptions): void;
export function finishRoutedExpertTrace(loader: object, requestId: string): RoutedExpertTrace;
export function isRoutedExpertTraceActive(loader: object): boolean;
export function cancelRoutedExpertTrace(loader: object): void;
export function readRoutedExpertSelection(loader: object, metadata: RoutedExpertSelection,
  read: () => Promise<ArrayBuffer>, required: boolean): Promise<ArrayBuffer | null>;
