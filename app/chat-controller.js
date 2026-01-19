

import { log } from '../src/debug/index.js';
import { formatChatMessages } from '../src/inference/pipeline/chat-format.js';

export class ChatController {
  #messages = [];

  #isGenerating = false;

  #abortController = null;

  #callbacks;

  #samplingParams = {
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    maxTokens: 512,
  };

  constructor(callbacks = {}) {
    this.#callbacks = callbacks;
  }

  get isGenerating() {
    return this.#isGenerating;
  }

  get messages() {
    return [...this.#messages];
  }

  setSamplingParams(params) {
    Object.assign(this.#samplingParams, params);
  }

  getSamplingParams() {
    return { ...this.#samplingParams };
  }

  async generate(message, pipeline) {
    if (!pipeline) {
      throw new Error('Pipeline not initialized');
    }

    if (this.#isGenerating) {
      throw new Error('Generation already in progress');
    }

    log.debug('ChatController', 'Generating response...');
    this.#isGenerating = true;
    this.#abortController = new AbortController();

    // Add user message
    this.#messages.push({ role: 'user', content: message });
    this.#callbacks.onUserMessage?.(message);

    // Format prompt with chat template
    const runtimeChat = pipeline.runtimeConfig?.inference?.chatTemplate;
    const chatTemplateEnabled = runtimeChat?.enabled ?? pipeline.modelConfig?.chatTemplateEnabled ?? false;
    const chatTemplateType = pipeline.modelConfig?.chatTemplateType ?? null;
    const prompt = formatChatMessages(
    this.#messages,
    chatTemplateEnabled ? chatTemplateType : null
    );

    this.#callbacks.onGenerationStart?.();

    let responseText = '';
    let responseCommitted = false;

    try {
      let tokenCount = 0;
      const startTime = performance.now();

      for await (const token of pipeline.generate(prompt, {
        maxTokens: this.#samplingParams.maxTokens,
        temperature: this.#samplingParams.temperature,
        topP: this.#samplingParams.topP,
        topK: this.#samplingParams.topK,
        useChatTemplate: false,
        signal: this.#abortController.signal,
      })) {
        if (this.#abortController.signal.aborted) break;

        this.#callbacks.onToken?.(token);
        tokenCount++;
        responseText += token;

        // Update TPS periodically
        if (tokenCount % 10 === 0) {
          const elapsed = (performance.now() - startTime) / 1000;
          this.#callbacks.onStats?.({ tokensPerSec: tokenCount / elapsed });
        }
      }

      if (responseText) {
        this.#messages.push({ role: 'assistant', content: responseText });
        responseCommitted = true;
      }

      const totalTime = (performance.now() - startTime) / 1000;
      this.#callbacks.onGenerationComplete?.({
        text: responseText,
        tokenCount,
        tokensPerSec: tokenCount / totalTime,
      });

      return responseText;
    } catch (error) {
      if (error.name === 'AbortError') {
        if (responseText && !responseCommitted) {
          this.#messages.push({ role: 'assistant', content: responseText });
        }
        this.#callbacks.onGenerationAborted?.(responseText);
        return responseText;
      } else {
        if (responseText && !responseCommitted) {
          this.#messages.push({ role: 'assistant', content: responseText });
        }
        log.error('ChatController', 'Generation error:', error);
        this.#callbacks.onGenerationError?.(error);
        throw error;
      }
    } finally {
      this.#isGenerating = false;
      this.#abortController = null;
    }
  }

  stop() {
    if (this.#abortController) {
      this.#abortController.abort();
    }
  }

  clear(pipeline) {
    if (pipeline && typeof pipeline.clearKVCache === 'function') {
      pipeline.clearKVCache();
    }
    this.#messages = [];
    log.debug('ChatController', 'Conversation cleared');
  }
}




