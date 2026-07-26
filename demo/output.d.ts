export declare function renderChatMessages(messages: Array<{ role: string; content: string }>): void;
export declare function beginChatTurn(messages: Array<{ role: string; content: string }>): void;
export declare function renderImportedChat(output: string, prompt?: string | null): void;
export declare function setPhase(label: string): void;
export declare function setTokSec(value: number | null): void;
export declare function clearTokSec(): void;
export declare function setPrefillProgress(percent: number): void;
export declare function appendToken(text: string): void;
export declare function clearOutput(): void;
export declare function showWordQuality(show: boolean): void;
export declare function renderWordQuality(quality: {
  words?: Array<{
    text: string;
    summedSurprisal: number | null;
    rollingPerplexity: number | null;
    cumulativePerplexity: number | null;
    tokenCount: number;
    rollingWindow: { unit: 'words' | 'tokens'; size: number };
  }>;
}): void;
export declare function setFinalStats(stats: Record<string, number | null | undefined>): void;
