/**
 * chat-ui.d.ts - Chat Interface Component Type Declarations
 * Agent-D | Phase 2 | app/
 *
 * Handles chat message display, streaming tokens, and user input.
 *
 * @module app/chat-ui
 */

/**
 * Message statistics
 */
export interface MessageStats {
  /** Number of tokens generated */
  tokens: number;
  /** Generation time in milliseconds */
  timeMs: number;
  /** Tokens per second */
  tokensPerSec: number;
}

/**
 * Chat UI callback functions
 */
export interface ChatUICallbacks {
  /** Called when user sends a message */
  onSend?: (message: string) => void;
  /** Called when stop is clicked */
  onStop?: () => void;
  /** Called when clear is clicked */
  onClear?: () => void;
}

/**
 * Message role type
 */
export type MessageRole = 'user' | 'assistant';

/**
 * Chat UI class for managing chat interface
 */
export declare class ChatUI {
  /**
   * @param container - Container element for chat
   * @param callbacks - Event callbacks
   */
  constructor(container: HTMLElement, callbacks?: ChatUICallbacks);

  /**
   * Enable or disable input
   */
  setInputEnabled(enabled: boolean): void;

  /**
   * Set loading state (waiting for model response)
   */
  setLoading(loading: boolean): void;

  /**
   * Add a complete message to the chat
   * @param role - Message role
   * @param content - Message content
   * @param stats - Optional generation stats
   */
  addMessage(role: MessageRole, content: string, stats?: MessageStats): void;

  /**
   * Start streaming a new assistant message
   */
  startStream(): void;

  /**
   * Append a token to the current stream
   * @param token - Token text
   */
  streamToken(token: string): void;

  /**
   * Finish the current stream
   */
  finishStream(): MessageStats;

  /**
   * Cancel the current stream
   */
  cancelStream(): void;

  /**
   * Clear all messages
   */
  clear(): void;

  /**
   * Focus the input field
   */
  focusInput(): void;

  /**
   * Check if currently streaming
   */
  isCurrentlyStreaming(): boolean;

  /**
   * Get current stream token count
   */
  getCurrentTokenCount(): number;
}

export default ChatUI;
