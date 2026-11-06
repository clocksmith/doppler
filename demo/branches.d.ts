export interface BranchCandidate {
  tokenId: number;
  text?: string;
  probability?: number | null;
}

export interface BranchTokenRecord {
  index: number;
  tokenId: number;
  text?: string;
  probability?: number | null;
  surprisal?: number | null;
  topCandidates?: BranchCandidate[];
  chosen?: boolean;
}

export type BranchStatus = 'running' | 'complete' | 'stopped';

export interface BranchNode {
  id: number;
  parentId: number | null;
  forkIndex: number;
  pathIds: number[];
  records: BranchTokenRecord[];
  status: BranchStatus;
  /** The completed run that produced this branch, owned by the demo core. */
  run?: unknown;
}

export interface BranchTree {
  inputTokenIds: number[];
  nodes: Map<number, BranchNode>;
  activeId: number;
  nextId: number;
}

export interface TokenFragmentGroup {
  start: number;
  end: number;
  text: string;
}

export function createBranchTree(input: {
  inputTokenIds: number[];
  tokenIds: number[];
  records: BranchTokenRecord[];
}): BranchTree;
export function activeBranch(tree: BranchTree): BranchNode;
export function branchChildren(tree: BranchTree, parentId: number, forkIndex?: number | null): BranchNode[];
export function selectBranch(tree: BranchTree, id: number): BranchNode;
export function startBranch(tree: BranchTree, forkIndex: number, tokenId: number): { node: BranchNode; created: boolean };
export function branchInputTokenIds(tree: BranchTree, node: BranchNode): number[];
export function appendBranchToken(node: BranchNode, tokenId: number, record?: BranchTokenRecord | null): void;
export function settleBranch(
  node: BranchNode,
  status: BranchStatus,
  generatedTokenIds?: number[] | null,
  records?: BranchTokenRecord[] | null
): BranchNode;
export function removeBranch(tree: BranchTree, node: BranchNode): void;
export function forkGroups(tree: BranchTree): Map<number, { members: number[]; active: number }>;
export function groupTokenFragments(
  records: BranchTokenRecord[],
  decodeTokenIds?: ((tokenIds: number[]) => string) | null
): TokenFragmentGroup[];
