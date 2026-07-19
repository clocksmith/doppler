export interface SourceTensorRenameRule {
  kind: 'rename';
  match: string;
  replace: string;
  expectedMatches: number;
  role?: string;
  group?: string;
}

export interface SourceTensorSplitPart {
  replace: string;
  size: number;
  role?: string;
  group?: string;
}

export interface SourceTensorSplitRule {
  kind: 'split';
  match: string;
  axis: 0;
  parts: SourceTensorSplitPart[];
  expectedMatches: number;
}

export interface SourceTensorIgnoreRule {
  kind: 'ignore';
  match: string;
  reason: string;
  expectedMatches: number;
}

export interface SourceTensorPolicy {
  requireAll: boolean;
  rules: Array<SourceTensorRenameRule | SourceTensorSplitRule | SourceTensorIgnoreRule>;
}

export interface SourceTensorDescriptor {
  name: string;
  shape: number[];
  offset: number;
  size: number;
  role?: string;
  group?: string;
  [key: string]: unknown;
}

export declare function applySourceTensorRules<T extends SourceTensorDescriptor>(
  tensors: T[],
  policy: SourceTensorPolicy | null
): T[];
