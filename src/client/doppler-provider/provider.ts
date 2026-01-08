/**
 * DOPPLER Provider
 * Main provider class that orchestrates model management and generation.
 * Implements the DopplerProviderInterface for llm-client.js integration.
 */

import type { KVCacheSnapshot } from '../../inference/pipeline.js';
import type { LoRAManifest } from '../../adapters/lora-loader.js';
import type { RDRRManifest } from '../../storage/rdrr-format.js';
import {
  DopplerCapabilities,
  type DopplerCapabilitiesType,
  type DopplerProviderInterface,
  type LoadProgressEvent,
  type GenerateOptions,
  type ChatMessage,
  type ChatResponse,
} from './types.js';
import {
  initDoppler,
  loadModel,
  unloadModel,
  loadLoRAAdapter,
  unloadLoRAAdapter,
  getActiveLoRA,
  getAvailableModels,
  destroyDoppler,
} from './model-manager.js';
import {
  generate,
  prefillKV,
  generateWithPrefixKV,
  formatChatMessages,
  dopplerChat,
} from './generation.js';

/**
 * Provider definition for llm-client.js
 */
export const DopplerProvider: DopplerProviderInterface = {
  name: 'doppler',
  displayName: 'DOPPLER',
  isLocal: true,

  async init(): Promise<boolean> {
    return initDoppler();
  },

  async loadModel(
    modelId: string,
    modelUrl?: string | null,
    onProgress?: ((event: LoadProgressEvent) => void) | null,
    localPath?: string | null
  ): Promise<boolean> {
    return loadModel(modelId, modelUrl ?? null, onProgress ?? null, localPath ?? null);
  },

  async chat(messages: ChatMessage[], options?: GenerateOptions): Promise<ChatResponse> {
    return dopplerChat(messages, options);
  },

  async *stream(messages: ChatMessage[], options?: GenerateOptions): AsyncGenerator<string> {
    const prompt = formatChatMessages(messages);
    for await (const token of generate(prompt, options)) {
      yield token;
    }
  },

  async prefillKV(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot> {
    return prefillKV(prompt, options);
  },

  async *generateWithPrefixKV(
    prefix: KVCacheSnapshot,
    prompt: string,
    options?: GenerateOptions
  ): AsyncGenerator<string> {
    for await (const token of generateWithPrefixKV(prefix, prompt, options)) {
      yield token;
    }
  },

  async loadLoRAAdapter(adapter: LoRAManifest | RDRRManifest | string): Promise<void> {
    return loadLoRAAdapter(adapter);
  },

  async unloadLoRAAdapter(): Promise<void> {
    return unloadLoRAAdapter();
  },

  getActiveLoRA(): string | null {
    return getActiveLoRA();
  },

  getCapabilities(): DopplerCapabilitiesType {
    return DopplerCapabilities;
  },

  async getModels(): Promise<string[]> {
    return getAvailableModels();
  },

  async destroy(): Promise<void> {
    return destroyDoppler();
  },
};

export default DopplerProvider;
