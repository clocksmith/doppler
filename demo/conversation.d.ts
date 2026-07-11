export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ConversationRequest {
  promptInput: { messages: Array<{ role: string; content: unknown }> };
  messages: ConversationMessage[];
  currentPrompt: string;
  historyEnabled: boolean;
  turnLimit: number;
  priorTurnCount: number;
}

export declare const DEFAULT_HISTORY_TURN_LIMIT: number;
export declare const MAX_STORED_HISTORY_TURNS: number;
export declare function normalizeHistoryTurnLimit(value: unknown, fallback?: number): number;
export declare function countConversationTurns(history: unknown): number;
export declare function trimConversationHistory(history: unknown, turnLimit?: number): ConversationMessage[];
export declare function createConversationRequest(
  history: unknown,
  prompt: string,
  options?: {
    historyEnabled?: boolean;
    turnLimit?: number;
    templateType?: string | null;
    translation?: { sourceLangCode?: string; targetLangCode?: string };
  }
): ConversationRequest;
export declare function appendConversationTurn(
  history: unknown,
  request: ConversationRequest,
  output: string
): ConversationMessage[];
