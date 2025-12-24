/**
 * chat-ui.ts - Chat Interface Component
 * Agent-D | Phase 2 | app/
 *
 * Handles chat message display, streaming tokens, and user input.
 *
 * @module app/chat-ui
 */

// ============================================================================
// Types and Interfaces
// ============================================================================

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

// ============================================================================
// ChatUI Class
// ============================================================================

export class ChatUI {
  private container: HTMLElement;
  private messagesElement: HTMLElement;
  private welcomeElement: HTMLElement | null;
  private inputElement: HTMLTextAreaElement;
  private sendBtn: HTMLButtonElement;
  private stopBtn: HTMLButtonElement;
  private clearBtn: HTMLButtonElement;

  private onSend: (message: string) => void;
  private onStop: () => void;
  private onClear: () => void;

  private currentStreamElement: HTMLElement | null = null;
  private isStreaming = false;
  private streamStartTime = 0;
  private streamTokenCount = 0;

  /**
   * @param container - Container element for chat
   * @param callbacks - Event callbacks
   */
  constructor(container: HTMLElement, callbacks: ChatUICallbacks = {}) {
    this.container = container;
    this.messagesElement = container.querySelector('#chat-messages') as HTMLElement;
    this.welcomeElement = container.querySelector('#welcome-message');
    this.inputElement = container.querySelector('#chat-input') as HTMLTextAreaElement;
    this.sendBtn = container.querySelector('#send-btn') as HTMLButtonElement;
    this.stopBtn = container.querySelector('#stop-btn') as HTMLButtonElement;
    this.clearBtn = container.querySelector('#clear-btn') as HTMLButtonElement;

    this.onSend = callbacks.onSend || (() => {});
    this.onStop = callbacks.onStop || (() => {});
    this.onClear = callbacks.onClear || (() => {});

    this._bindEvents();
  }

  /**
   * Bind DOM event listeners
   */
  private _bindEvents(): void {
    // Auto-resize textarea
    this.inputElement.addEventListener('input', () => {
      this.inputElement.style.height = 'auto';
      this.inputElement.style.height = Math.min(this.inputElement.scrollHeight, 150) + 'px';
    });

    // Send on Enter (Shift+Enter for newline)
    this.inputElement.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this._handleSend();
      }
    });

    this.sendBtn.addEventListener('click', () => this._handleSend());
    this.stopBtn.addEventListener('click', () => this.onStop());
    this.clearBtn.addEventListener('click', () => {
      this.clear();
      this.onClear();
    });
  }

  /**
   * Handle send action
   */
  private _handleSend(): void {
    const message = this.inputElement.value.trim();
    if (message && !this.isStreaming) {
      this.inputElement.value = '';
      this.inputElement.style.height = 'auto';
      this.onSend(message);
    }
  }

  /**
   * Enable or disable input
   */
  setInputEnabled(enabled: boolean): void {
    this.inputElement.disabled = !enabled;
    this.sendBtn.disabled = !enabled;
  }

  /**
   * Set loading state (waiting for model response)
   */
  setLoading(loading: boolean): void {
    if (loading) {
      this.setInputEnabled(false);
      this.stopBtn.hidden = false;
    } else {
      this.setInputEnabled(true);
      this.stopBtn.hidden = true;
    }
  }

  /**
   * Add a complete message to the chat
   * @param role - Message role
   * @param content - Message content
   * @param stats - Optional generation stats
   */
  addMessage(role: MessageRole, content: string, stats?: MessageStats): void {
    this._hideWelcome();

    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;

    let statsHtml = '';
    if (stats) {
      statsHtml = `
        <div class="message-stats">
          ${stats.tokens} tokens · ${(stats.timeMs / 1000).toFixed(1)}s · ${stats.tokensPerSec.toFixed(1)} tok/s
        </div>
      `;
    }

    messageEl.innerHTML = `
      <div class="message-role">${role}</div>
      <div class="message-content">${this._escapeHtml(content)}</div>
      ${statsHtml}
    `;

    this.messagesElement.appendChild(messageEl);
    this._scrollToBottom();
  }

  /**
   * Start streaming a new assistant message
   */
  startStream(): void {
    this._hideWelcome();
    this.isStreaming = true;
    this.streamStartTime = performance.now();
    this.streamTokenCount = 0;

    this.currentStreamElement = document.createElement('div');
    this.currentStreamElement.className = 'message assistant';
    this.currentStreamElement.innerHTML = `
      <div class="message-role">assistant</div>
      <div class="message-content"><span class="cursor"></span></div>
      <div class="message-stats"></div>
    `;

    this.messagesElement.appendChild(this.currentStreamElement);
    this._scrollToBottom();
    this.setLoading(true);
  }

  /**
   * Append a token to the current stream
   * @param token - Token text
   */
  streamToken(token: string): void {
    if (!this.currentStreamElement) return;

    this.streamTokenCount++;
    const contentEl = this.currentStreamElement.querySelector('.message-content')!;
    const cursor = contentEl.querySelector('.cursor')!;

    // Insert token before cursor
    const textNode = document.createTextNode(token);
    contentEl.insertBefore(textNode, cursor);

    // Update live stats
    const elapsed = performance.now() - this.streamStartTime;
    const tps = this.streamTokenCount / (elapsed / 1000);
    const statsEl = this.currentStreamElement.querySelector('.message-stats')!;
    statsEl.textContent = `${this.streamTokenCount} tokens · ${(elapsed / 1000).toFixed(1)}s · ${tps.toFixed(1)} tok/s`;

    this._scrollToBottom();
  }

  /**
   * Finish the current stream
   */
  finishStream(): MessageStats {
    if (!this.currentStreamElement) {
      return { tokens: 0, timeMs: 0, tokensPerSec: 0 };
    }

    const elapsed = performance.now() - this.streamStartTime;
    const tps = this.streamTokenCount / (elapsed / 1000);

    // Remove cursor
    const cursor = this.currentStreamElement.querySelector('.cursor');
    if (cursor) {
      cursor.remove();
    }

    // Final stats
    const statsEl = this.currentStreamElement.querySelector('.message-stats')!;
    statsEl.textContent = `${this.streamTokenCount} tokens · ${(elapsed / 1000).toFixed(1)}s · ${tps.toFixed(1)} tok/s`;

    this.currentStreamElement = null;
    this.isStreaming = false;
    this.setLoading(false);

    return {
      tokens: this.streamTokenCount,
      timeMs: elapsed,
      tokensPerSec: tps,
    };
  }

  /**
   * Cancel the current stream
   */
  cancelStream(): void {
    if (this.currentStreamElement) {
      const cursor = this.currentStreamElement.querySelector('.cursor');
      if (cursor) {
        cursor.remove();
      }

      // Add cancelled indicator
      const contentEl = this.currentStreamElement.querySelector('.message-content')!;
      contentEl.innerHTML += '<span style="color: var(--text-muted);"> [stopped]</span>';

      const statsEl = this.currentStreamElement.querySelector('.message-stats')!;
      const elapsed = performance.now() - this.streamStartTime;
      statsEl.textContent = `${this.streamTokenCount} tokens · ${(elapsed / 1000).toFixed(1)}s (stopped)`;
    }

    this.currentStreamElement = null;
    this.isStreaming = false;
    this.setLoading(false);
  }

  /**
   * Clear all messages
   */
  clear(): void {
    this.messagesElement.innerHTML = '';
    if (this.welcomeElement) {
      this.messagesElement.appendChild(this.welcomeElement);
      this.welcomeElement.hidden = false;
    }
    this.currentStreamElement = null;
    this.isStreaming = false;
    this.setLoading(false);
  }

  /**
   * Hide welcome message
   */
  private _hideWelcome(): void {
    if (this.welcomeElement) {
      this.welcomeElement.hidden = true;
    }
  }

  /**
   * Scroll to bottom of messages
   */
  private _scrollToBottom(): void {
    this.messagesElement.scrollTop = this.messagesElement.scrollHeight;
  }

  /**
   * Escape HTML to prevent XSS
   */
  private _escapeHtml(str: string): string {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  /**
   * Focus the input field
   */
  focusInput(): void {
    this.inputElement.focus();
  }

  /**
   * Check if currently streaming
   */
  isCurrentlyStreaming(): boolean {
    return this.isStreaming;
  }

  /**
   * Get current stream token count
   */
  getCurrentTokenCount(): number {
    return this.streamTokenCount;
  }
}

export default ChatUI;
