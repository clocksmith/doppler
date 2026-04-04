export type ChatRole = 'system' | 'user' | 'assistant';

export interface TranslateGemmaTextContent {
  type: 'text';
  source_lang_code: string;
  target_lang_code: string;
  text: string;
}

export interface TranslateGemmaImageContent {
  type: 'image';
  source_lang_code: string;
  target_lang_code: string;
  image?: string;
}

export type ChatContentBlock =
  | TranslateGemmaTextContent
  | TranslateGemmaImageContent
  | Record<string, unknown>;

export type ChatMessageContent = string | ChatContentBlock[];

export interface ChatMessage {
  role: ChatRole;
  content: ChatMessageContent;
}

export type ChatTemplateType =
  | 'gemma'
  | 'gemma4'
  | 'llama3'
  | 'gpt-oss'
  | 'chatml'
  | 'qwen'
  | 'translategemma'
  | null;

export declare function formatGemmaChat(messages: ChatMessage[]): string;

export interface ChatFormatOptions {
  thinking?: boolean;
}

export declare function formatGemma4Chat(messages: ChatMessage[], options?: ChatFormatOptions): string;

export declare function formatLlama3Chat(messages: ChatMessage[]): string;

export declare function formatGptOssChat(messages: ChatMessage[]): string;

export declare function formatTranslateGemmaChat(messages: ChatMessage[]): string;

export declare function formatChatMessages(messages: ChatMessage[], templateType?: ChatTemplateType, options?: ChatFormatOptions): string;
