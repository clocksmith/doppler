export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ConversationRequest {
  promptInput: { messages: Array<{ role: string; content: unknown }> };
  messages: ConversationMessage[];
  contextMessages: ConversationMessage[];
  currentPrompt: string;
  priorTurnCount: number;
}

export declare const MAX_CONTEXT_HISTORY_TURNS: number;
export declare function countConversationTurns(history: unknown): number;
export declare function normalizeConversationHistory(history: unknown): ConversationMessage[];
export declare function createConversationRequest(
  history: unknown,
  prompt: string,
  options?: {
    templateType?: string | null;
    translation?: { sourceLangCode?: string; targetLangCode?: string };
  }
): ConversationRequest;
export declare function appendConversationTurn(
  history: unknown,
  request: ConversationRequest,
  output: string
): ConversationMessage[];
