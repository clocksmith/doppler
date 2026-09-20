export function getPrompt(): any;
export function setPromptValue(value: any): void;
export function clearPrompt(): void;
export function buildConversationRequest(prompt: any, options?: {}): import("./conversation.js").ConversationRequest;
export function recordConversationTurn(request: any, output: any, options?: {
    render?: boolean;
    quality?: import('./ui/chat-markdown.js').ChatMarkdownQuality | null;
}): void;
export function clearConversation(): Promise<void>;
export function clearConversationHistory(): void;
export function restoreConversationHistory(messages: any): void;
export function resetConversationForModel(modelId: any): void;
export function setRunHandler(handler: any): void;
export function isSendReady({ model, pipeline, prompt, generating, prefilling }: {
    model: any;
    pipeline: any;
    prompt: any;
    generating?: boolean | undefined;
    prefilling?: boolean | undefined;
}): boolean;
export function syncSendButton(options?: {}): boolean;
export function initInput(): Promise<void>;
export function setGenerating(active: any): void;
