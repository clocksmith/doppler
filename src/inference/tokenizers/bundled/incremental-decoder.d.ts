export interface IncrementalTokenDecoder {
  push(tokenId: number): string;
  pendingText(): string;
  finish(): string;
}
export function createBundledIncrementalDecoder(options: {
  tokenForId(id: number): string | undefined;
  byteLevel: boolean;
  byteDecoder: Map<string, number>;
  wordPiece: boolean;
  splitEveryCharacter: boolean;
  wordPiecePrefix: string;
}): IncrementalTokenDecoder;
