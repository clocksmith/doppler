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

export interface ChatFormatOptions {
  thinking?: boolean;
}

export declare function formatChatMessages(messages: ChatMessage[], templateType?: ChatTemplateType, options?: ChatFormatOptions): string;
