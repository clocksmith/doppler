

// ============================================================================
// ChatUI Class
// ============================================================================

export class ChatUI {
  
  #container;
  
  #messagesElement;
  
  #welcomeElement;
  
  #inputElement;
  
  #sendBtn;
  
  #stopBtn;
  
  #clearBtn;

  
  #onSend;
  
  #onStop;
  
  #onClear;

  
  #currentStreamElement = null;
  
  #isStreaming = false;
  
  #streamStartTime = 0;
  
  #streamTokenCount = 0;

  
  constructor(container, callbacks = {}) {
    this.#container = container;
    this.#messagesElement =  (container.querySelector('#chat-messages'));
    this.#welcomeElement = container.querySelector('#welcome-message');
    this.#inputElement =  (container.querySelector('#chat-input'));
    this.#sendBtn =  (container.querySelector('#send-btn'));
    this.#stopBtn =  (container.querySelector('#stop-btn'));
    this.#clearBtn =  (container.querySelector('#clear-btn'));

    this.#onSend = callbacks.onSend || (() => {});
    this.#onStop = callbacks.onStop || (() => {});
    this.#onClear = callbacks.onClear || (() => {});

    this.#bindEvents();
  }

  
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

  
  #handleSend() {
    const message = this.#inputElement.value.trim();
    if (message && !this.#isStreaming) {
      this.#inputElement.value = '';
      this.#inputElement.style.height = 'auto';
      this.#onSend(message);
    }
  }

  
  setInputEnabled(enabled) {
    this.#inputElement.disabled = !enabled;
    this.#sendBtn.disabled = !enabled;
  }

  
  setLoading(loading) {
    if (loading) {
      this.setInputEnabled(false);
      this.#stopBtn.hidden = false;
    } else {
      this.setInputEnabled(true);
      this.#stopBtn.hidden = true;
    }
  }

  
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

  
  streamToken(token) {
    if (!this.#currentStreamElement) return;

    this.#streamTokenCount++;
    const contentEl =  (this.#currentStreamElement.querySelector('.message-content'));
    const cursor =  (contentEl.querySelector('.cursor'));

    // Insert token before cursor
    const textNode = document.createTextNode(token);
    contentEl.insertBefore(textNode, cursor);

    // Update live stats
    const elapsed = performance.now() - this.#streamStartTime;
    const tps = this.#streamTokenCount / (elapsed / 1000);
    const statsEl =  (this.#currentStreamElement.querySelector('.message-stats'));
    statsEl.textContent = `${this.#streamTokenCount} tokens · ${(elapsed / 1000).toFixed(1)}s · ${tps.toFixed(1)} tok/s`;

    this.#scrollToBottom();
  }

  
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
    const statsEl =  (this.#currentStreamElement.querySelector('.message-stats'));
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

  
  cancelStream() {
    if (this.#currentStreamElement) {
      const cursor = this.#currentStreamElement.querySelector('.cursor');
      if (cursor) {
        cursor.remove();
      }

      // Add cancelled indicator
      const contentEl =  (this.#currentStreamElement.querySelector('.message-content'));
      contentEl.innerHTML += '<span class="muted"> [stopped]</span>';

      const statsEl =  (this.#currentStreamElement.querySelector('.message-stats'));
      const elapsed = performance.now() - this.#streamStartTime;
      statsEl.textContent = `${this.#streamTokenCount} tokens · ${(elapsed / 1000).toFixed(1)}s (stopped)`;
    }

    this.#currentStreamElement = null;
    this.#isStreaming = false;
    this.setLoading(false);
  }

  
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

  
  #hideWelcome() {
    if (this.#welcomeElement) {
      this.#welcomeElement.hidden = true;
    }
  }

  
  #scrollToBottom() {
    this.#messagesElement.scrollTop = this.#messagesElement.scrollHeight;
  }

  
  #escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  
  focusInput() {
    this.#inputElement.focus();
  }

  
  isCurrentlyStreaming() {
    return this.#isStreaming;
  }

  
  getCurrentTokenCount() {
    return this.#streamTokenCount;
  }
}

export default ChatUI;
