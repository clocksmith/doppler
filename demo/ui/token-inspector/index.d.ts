export interface InspectedToken {
  text: string;
  tokenId?: number | null;
  probability?: number | null;
  surprisal?: number | null;
  topCandidates?: Array<{ text: string; tokenId: number; probability: number; logit?: number }>;
}
export declare function isTokenInspectorActive(): boolean;
export declare function setTokenInspectorActive(active: boolean): void;
export declare function selectToken(index: number): void;
export declare function renderTokenInspector(tokens: InspectedToken[], streamContainer: HTMLElement, cardContainer: HTMLElement): void;
