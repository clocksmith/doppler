export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export type ChatTemplateType = 'gemma' | 'llama3' | 'gpt-oss' | null;

export declare function formatGemmaChat(messages: ChatMessage[]): string;

export declare function formatLlama3Chat(messages: ChatMessage[]): string;

export declare function formatGptOssChat(messages: ChatMessage[]): string;

export declare function formatChatMessages(
  messages: ChatMessage[],
  templateType?: ChatTemplateType
): string;
