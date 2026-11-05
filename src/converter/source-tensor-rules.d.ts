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

/**
 * Keep matched tensors in their pinned source files: conversion links each file
 * into the artifact as one shard and records each tensor's byte span there.
 */
export interface SourceTensorRetainRule {
  kind: 'retain';
  match: string;
  reason: string;
  role: string;
  /** Group of the retained tensors; must not be a layer group, so layer prefetch never reads them. */
  group: string;
  /** SHA-256 (lowercase hex) of each complete source file holding a matched tensor. */
  files: Record<string, string>;
  expectedMatches: number;
}

/** Set on tensors matched by a retain rule. */
export interface RetainedSourceBinding {
  file: string;
  sha256: string;
}

export interface SourceTensorPolicy {
  requireAll: boolean;
  rules: Array<SourceTensorRenameRule | SourceTensorSplitRule | SourceTensorIgnoreRule | SourceTensorRetainRule>;
}

export interface SourceTensorDescriptor {
  name: string;
  shape: number[];
  offset: number;
  size: number;
  role?: string;
  group?: string;
  sourcePath?: string;
  retainedSource?: RetainedSourceBinding;
  [key: string]: unknown;
}

export declare function applySourceTensorRules<T extends SourceTensorDescriptor>(
  tensors: T[],
  policy: SourceTensorPolicy | null
): T[];
