export interface StructuredPromptRequest {
  messages: unknown[];
}

export declare function resolvePromptInput(
  state: Record<string, unknown>,
  prompt: string | unknown[] | StructuredPromptRequest,
  useChatTemplate: boolean,
  contextLabel: string
): string;
