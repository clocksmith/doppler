export interface InspectedToken {
  text: string;
  tokenId?: number | null;
  probability?: number | null;
  surprisal?: number | null;
  topCandidates?: Array<{ text: string; tokenId: number; probability: number; logit?: number }>;
  chosen?: boolean;
}
export interface TokenInspectorOptions {
  /** Decodes grouped fragments together so split Unicode reads correctly. */
  decodeTokenIds?: ((tokenIds: number[]) => string) | null;
  /** Fork switchers on the shown path, keyed by token position. */
  forks?: Map<number, { members: number[]; active: number }>;
  branchable?: boolean;
  unavailableReason?: string;
  onBranch?: ((index: number, tokenId: number) => void) | null;
  onSwitch?: ((branchId: number) => void) | null;
}
export declare function isTokenInspectorActive(): boolean;
/** Recorded tokens of the shown answer. */
export declare function getTokenInspectorTokenCount(): number;
export declare function setTokenInspectorActive(active: boolean): void;
export declare function setTokenInspectorOptions(options?: TokenInspectorOptions): void;
export declare function getTokenInspectorSelection(): number;
export declare function selectToken(index: number, options?: { focus?: boolean }): void;
export declare function appendTokenInspectorToken(token: InspectedToken, streamContainer?: HTMLElement | null, cardContainer?: HTMLElement | null): void;
export declare function renderTokenInspector(tokens: InspectedToken[], streamContainer: HTMLElement, cardContainer: HTMLElement): void;
export declare function updateTokenInspector(tokens: InspectedToken[]): void;
