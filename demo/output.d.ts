export declare function renderChatMessages(messages: Array<{ role: string; content: string }>, annotations?: Array<{
  text: string;
  quality: import('./ui/chat-markdown.js').ChatMarkdownQuality | null;
}>): void;
export declare function beginChatTurn(messages: Array<{ role: string; content: string }>): void;
export declare function renderImportedChat(output: string, prompt?: string | null): void;
export declare function setPhase(label: string): void;
export declare function clearTokSec(): void;
export declare function setPrefillProgress(percent: number): void;
export declare function createOutputStream(
  decodeTokenIds: (tokenIds: number[]) => string,
  signal?: AbortSignal
): {
  push(tokenId: number, token?: import('./ui/token-inspector/index.js').InspectedToken | null): void;
  finish(finalText?: string): string;
};
export declare function clearOutput(): void;
export declare function showTokenInspectorView(show: boolean): void;
export declare function renderTokenInspection(tokens: import('./ui/token-inspector/index.js').InspectedToken[]): void;
export declare function showWordQuality(show: boolean): void;
export declare function renderWordQuality(quality: import('./ui/chat-markdown.js').ChatMarkdownQuality, text?: string): void;
export declare function setFinalStats(stats: Record<string, number | null | undefined>): void;
