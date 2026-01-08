/**
 * chat-ui.js - Chat Interface Component
 * Agent-D | Phase 2 | app/
 *
 * Handles chat message display, streaming tokens, and user input.
 *
 * @module app/chat-ui
 */

// ============================================================================
// ChatUI Class
// ============================================================================

export class ChatUI {
  /** @type {HTMLElement} */
  #container;
  /** @type {HTMLElement} */
  #messagesElement;
  /** @type {HTMLElement | null} */
  #welcomeElement;
  /** @type {HTMLTextAreaElement} */
  #inputElement;
  /** @type {HTMLButtonElement} */
  #sendBtn;
  /** @type {HTMLButtonElement} */
  #stopBtn;
  /** @type {HTMLButtonElement} */
  #clearBtn;

  /** @type {(message: string) => void} */
  #onSend;
  /** @type {() => void} */
  #onStop;
  /** @type {() => void} */
  #onClear;

  /** @type {HTMLElement | null} */
  #currentStreamElement = null;
  /** @type {boolean} */
  #isStreaming = false;
  /** @type {number} */
  #streamStartTime = 0;
  /** @type {number} */
  #streamTokenCount = 0;

  /**
   * @param {HTMLElement} container - Container element for chat
   * @param {import('./chat-ui.js').ChatUICallbacks} [callbacks] - Event callbacks
   */
  constructor(container, callbacks = {}) {
    this.#container = container;
    this.#messagesElement = /** @type {HTMLElement} */ (container.querySelector('#chat-messages'));
    this.#welcomeElement = container.querySelector('#welcome-message');
    this.#inputElement = /** @type {HTMLTextAreaElement} */ (container.querySelector('#chat-input'));
    this.#sendBtn = /** @type {HTMLButtonElement} */ (container.querySelector('#send-btn'));
    this.#stopBtn = /** @type {HTMLButtonElement} */ (container.querySelector('#stop-btn'));
    this.#clearBtn = /** @type {HTMLButtonElement} */ (container.querySelector('#clear-btn'));

    this.#onSend = callbacks.onSend || (() => {});
    this.#onStop = callbacks.onStop || (() => {});
    this.#onClear = callbacks.onClear || (() => {});

    this.#bindEvents();
  }

  /**
   * Bind DOM event listeners
   */
  #bindEvents() {
    // Auto-resize textarea
    this.#inputElement.addEventListener('input', () => {
      this.#inputElement.style.height = 'auto';
      this.#inputElement.style.height = Math.min(this.#inputElement.scrollHeight, 150) + 'px';
    });

    // Send on Enter (Shift+Enter for newline)
    this.#inputElement.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.#handleSend();
      }
    });

    this.#sendBtn.addEventListener('click', () => this.#handleSend());
    this.#stopBtn.addEventListener('click', () => this.#onStop());
    this.#clearBtn.addEventListener('click', () => {
      this.clear();
      this.#onClear();
    });
  }

  /**
   * Handle send action
   */
  #handleSend() {
    const message = this.#inputElement.value.trim();
    if (message && !this.#isStreaming) {
      this.#inputElement.value = '';
      this.#inputElement.style.height = 'auto';
      this.#onSend(message);
    }
  }

  /**
   * Enable or disable input
   * @param {boolean} enabled
   */
  setInputEnabled(enabled) {
    this.#inputElement.disabled = !enabled;
    this.#sendBtn.disabled = !enabled;
  }

  /**
   * Set loading state (waiting for model response)
   * @param {boolean} loading
   */
  setLoading(loading) {
    if (loading) {
      this.setInputEnabled(false);
      this.#stopBtn.hidden = false;
    } else {
      this.setInputEnabled(true);
      this.#stopBtn.hidden = true;
    }
  }

  /**
   * Add a complete message to the chat
   * @param {import('./chat-ui.js').MessageRole} role - Message role
   * @param {string} content - Message content
   * @param {import('./chat-ui.js').MessageStats} [stats] - Optional generation stats
   */
  addMessage(role, content, stats) {
    this.#hideWelcome();

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
      <div class="message-content">${this.#escapeHtml(content)}</div>
      ${statsHtml}
    `;

    this.#messagesElement.appendChild(messageEl);
    this.#scrollToBottom();
  }

  /**
   * Start streaming a new assistant message
   */
  startStream() {
    this.#hideWelcome();
    this.#isStreaming = true;
    this.#streamStartTime = performance.now();
    this.#streamTokenCount = 0;

    this.#currentStreamElement = document.createElement('div');
    this.#currentStreamElement.className = 'message assistant';
    this.#currentStreamElement.innerHTML = `
      <div class="message-role">assistant</div>
      <div class="message-content"><span class="cursor"></span></div>
      <div class="message-stats"></div>
    `;

    this.#messagesElement.appendChild(this.#currentStreamElement);
    this.#scrollToBottom();
    this.setLoading(true);
  }

  /**
   * Append a token to the current stream
   * @param {string} token - Token text
   */
  streamToken(token) {
    if (!this.#currentStreamElement) return;

    this.#streamTokenCount++;
    const contentEl = /** @type {HTMLElement} */ (this.#currentStreamElement.querySelector('.message-content'));
    const cursor = /** @type {HTMLElement} */ (contentEl.querySelector('.cursor'));

    // Insert token before cursor
    const textNode = document.createTextNode(token);
    contentEl.insertBefore(textNode, cursor);

    // Update live stats
    const elapsed = performance.now() - this.#streamStartTime;
    const tps = this.#streamTokenCount / (elapsed / 1000);
    const statsEl = /** @type {HTMLElement} */ (this.#currentStreamElement.querySelector('.message-stats'));
    statsEl.textContent = `${this.#streamTokenCount} tokens · ${(elapsed / 1000).toFixed(1)}s · ${tps.toFixed(1)} tok/s`;

    this.#scrollToBottom();
  }

  /**
   * Finish the current stream
   * @returns {import('./chat-ui.js').MessageStats}
   */
  finishStream() {
    if (!this.#currentStreamElement) {
      return { tokens: 0, timeMs: 0, tokensPerSec: 0 };
    }

    const elapsed = performance.now() - this.#streamStartTime;
    const tps = this.#streamTokenCount / (elapsed / 1000);

    // Remove cursor
    const cursor = this.#currentStreamElement.querySelector('.cursor');
    if (cursor) {
      cursor.remove();
    }

    // Final stats
    const statsEl = /** @type {HTMLElement} */ (this.#currentStreamElement.querySelector('.message-stats'));
    statsEl.textContent = `${this.#streamTokenCount} tokens · ${(elapsed / 1000).toFixed(1)}s · ${tps.toFixed(1)} tok/s`;

    this.#currentStreamElement = null;
    this.#isStreaming = false;
    this.setLoading(false);

    return {
      tokens: this.#streamTokenCount,
      timeMs: elapsed,
      tokensPerSec: tps,
    };
  }

  /**
   * Cancel the current stream
   */
  cancelStream() {
    if (this.#currentStreamElement) {
      const cursor = this.#currentStreamElement.querySelector('.cursor');
      if (cursor) {
        cursor.remove();
      }

      // Add cancelled indicator
      const contentEl = /** @type {HTMLElement} */ (this.#currentStreamElement.querySelector('.message-content'));
      contentEl.innerHTML += '<span class="muted"> [stopped]</span>';

      const statsEl = /** @type {HTMLElement} */ (this.#currentStreamElement.querySelector('.message-stats'));
      const elapsed = performance.now() - this.#streamStartTime;
      statsEl.textContent = `${this.#streamTokenCount} tokens · ${(elapsed / 1000).toFixed(1)}s (stopped)`;
    }

    this.#currentStreamElement = null;
    this.#isStreaming = false;
    this.setLoading(false);
  }

  /**
   * Clear all messages
   */
  clear() {
    this.#messagesElement.innerHTML = '';
    if (this.#welcomeElement) {
      this.#messagesElement.appendChild(this.#welcomeElement);
      this.#welcomeElement.hidden = false;
    }
    this.#currentStreamElement = null;
    this.#isStreaming = false;
    this.setLoading(false);
  }

  /**
   * Hide welcome message
   */
  #hideWelcome() {
    if (this.#welcomeElement) {
      this.#welcomeElement.hidden = true;
    }
  }

  /**
   * Scroll to bottom of messages
   */
  #scrollToBottom() {
    this.#messagesElement.scrollTop = this.#messagesElement.scrollHeight;
  }

  /**
   * Escape HTML to prevent XSS
   * @param {string} str
   * @returns {string}
   */
  #escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  /**
   * Focus the input field
   */
  focusInput() {
    this.#inputElement.focus();
  }

  /**
   * Check if currently streaming
   * @returns {boolean}
   */
  isCurrentlyStreaming() {
    return this.#isStreaming;
  }

  /**
   * Get current stream token count
   * @returns {number}
   */
  getCurrentTokenCount() {
    return this.#streamTokenCount;
  }
}

export default ChatUI;
