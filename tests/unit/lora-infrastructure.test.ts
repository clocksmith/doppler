/**
 * LoRA Infrastructure Tests
 *
 * Tests for the complete LoRA adapter infrastructure:
 * - Adapter manifest validation
 * - LoRA weight loading
 * - Adapter manager enable/disable
 * - Adapter registry persistence
 *
 * @module tests/unit/lora-infrastructure
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  // Manifest
  validateManifest,
  parseManifest,
  serializeManifest,
  createManifest,
  computeLoRAScale,
  type AdapterManifest,
  // Loader
  loadLoRAFromManifest,
  type LoRAManifest,
  // Manager
  AdapterManager,
  resetAdapterManager,
  // Registry
  AdapterRegistry,
  createMemoryRegistry,
  resetAdapterRegistry,
} from '../../src/adapters/index.js';

// ============================================================================
// Test Fixtures
// ============================================================================

function createTestManifest(overrides: Partial<AdapterManifest> = {}): AdapterManifest {
  return {
    id: 'test-adapter',
    name: 'Test LoRA Adapter',
    version: '1.0.0',
    baseModel: 'gemma-3-1b',
    rank: 8,
    alpha: 16,
    targetModules: ['q_proj', 'v_proj'],
    ...overrides,
  };
}

function createTestLoRAManifest(): LoRAManifest {
  // Create small test tensors (8x4 for A, 4x8 for B with rank=4)
  const rank = 4;
  const hiddenSize = 8;

  // Layer 0, q_proj
  const aData = new Array(rank * hiddenSize).fill(0).map((_, i) => Math.sin(i * 0.1));
  const bData = new Array(hiddenSize * rank).fill(0).map((_, i) => Math.cos(i * 0.1));

  return {
    name: 'test-lora',
    version: '1.0.0',
    baseModel: 'test-model',
    rank,
    alpha: 8,
    targetModules: ['q_proj', 'v_proj'],
    tensors: [
      {
        name: 'layer.0.q_proj.lora_a',
        shape: [rank, hiddenSize],
        data: aData,
      },
      {
        name: 'layer.0.q_proj.lora_b',
        shape: [hiddenSize, rank],
        data: bData,
      },
      {
        name: 'layer.0.v_proj.lora_a',
        shape: [rank, hiddenSize],
        data: aData.map(x => x * 0.5),
      },
      {
        name: 'layer.0.v_proj.lora_b',
        shape: [hiddenSize, rank],
        data: bData.map(x => x * 0.5),
      },
    ],
  };
}

// ============================================================================
// Adapter Manifest Tests
// ============================================================================

describe('AdapterManifest', () => {
  describe('validateManifest', () => {
    it('should validate a correct manifest', () => {
      const manifest = createTestManifest();
      const result = validateManifest(manifest);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should reject manifest missing required fields', () => {
      const result = validateManifest({});
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
      expect(result.errors.some(e => e.field === 'id')).toBe(true);
      expect(result.errors.some(e => e.field === 'name')).toBe(true);
      expect(result.errors.some(e => e.field === 'baseModel')).toBe(true);
    });

    it('should reject invalid id format', () => {
      const manifest = createTestManifest({ id: 'invalid id with spaces' });
      const result = validateManifest(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.field === 'id')).toBe(true);
    });

    it('should reject invalid rank', () => {
      const manifest = createTestManifest({ rank: 0 });
      const result = validateManifest(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.field === 'rank')).toBe(true);
    });

    it('should reject invalid targetModules', () => {
      const manifest = createTestManifest({
        targetModules: ['invalid_module'] as any,
      });
      const result = validateManifest(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.field === 'targetModules')).toBe(true);
    });

    it('should validate semantic version format', () => {
      const manifest = createTestManifest({ version: 'not-a-version' });
      const result = validateManifest(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.field === 'version')).toBe(true);
    });

    it('should validate checksum format', () => {
      const manifest = createTestManifest({ checksum: 'short' });
      const result = validateManifest(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.field === 'checksum')).toBe(true);
    });
  });

  describe('parseManifest', () => {
    it('should parse valid JSON', () => {
      const manifest = createTestManifest();
      const json = JSON.stringify(manifest);
      const parsed = parseManifest(json);
      expect(parsed.id).toBe(manifest.id);
      expect(parsed.name).toBe(manifest.name);
    });

    it('should throw on invalid JSON', () => {
      expect(() => parseManifest('not json')).toThrow('Invalid JSON');
    });

    it('should throw on invalid manifest', () => {
      expect(() => parseManifest('{}')).toThrow('Manifest validation failed');
    });
  });

  describe('serializeManifest', () => {
    it('should serialize valid manifest', () => {
      const manifest = createTestManifest();
      const json = serializeManifest(manifest);
      expect(JSON.parse(json)).toEqual(manifest);
    });

    it('should throw on invalid manifest', () => {
      expect(() => serializeManifest({} as any)).toThrow('Cannot serialize invalid manifest');
    });

    it('should support pretty printing', () => {
      const manifest = createTestManifest();
      const pretty = serializeManifest(manifest, true);
      expect(pretty).toContain('\n');
    });
  });

  describe('createManifest', () => {
    it('should create manifest with defaults', () => {
      const manifest = createManifest({
        id: 'test',
        name: 'Test',
        baseModel: 'model',
        rank: 8,
        alpha: 16,
        targetModules: ['q_proj'],
      });
      expect(manifest.version).toBe('1.0.0');
      expect(manifest.checksumAlgorithm).toBe('sha256');
      expect(manifest.weightsFormat).toBe('safetensors');
    });

    it('should allow overriding defaults', () => {
      const manifest = createManifest({
        id: 'test',
        name: 'Test',
        baseModel: 'model',
        rank: 8,
        alpha: 16,
        targetModules: ['q_proj'],
        version: '2.0.0',
      });
      expect(manifest.version).toBe('2.0.0');
    });
  });

  describe('computeLoRAScale', () => {
    it('should compute scale as alpha/rank', () => {
      expect(computeLoRAScale(8, 16)).toBe(2);
      expect(computeLoRAScale(4, 8)).toBe(2);
      expect(computeLoRAScale(16, 8)).toBe(0.5);
    });

    it('should return 1 for rank 0', () => {
      expect(computeLoRAScale(0, 16)).toBe(1);
    });
  });
});

// ============================================================================
// LoRA Loader Tests
// ============================================================================

describe('LoRALoader', () => {
  describe('loadLoRAFromManifest', () => {
    it('should load adapter with inline tensor data', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);

      expect(adapter.name).toBe('test-lora');
      expect(adapter.rank).toBe(4);
      expect(adapter.alpha).toBe(8);
      expect(adapter.layers.size).toBe(1);
    });

    it('should parse tensor names correctly', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);

      const layer0 = adapter.layers.get(0);
      expect(layer0).toBeDefined();
      expect(layer0!['q_proj']).toBeDefined();
      expect(layer0!['v_proj']).toBeDefined();
    });

    it('should compute scale correctly', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);

      const layer0 = adapter.layers.get(0);
      expect(layer0!['q_proj'].scale).toBe(8 / 4); // alpha / rank
    });

    it('should validate tensor shapes', async () => {
      const manifest = createTestLoRAManifest();
      manifest.tensors[0].data = [1, 2, 3]; // Wrong size
      manifest.tensors[0].shape = [2, 2]; // But shape says 4 elements

      await expect(loadLoRAFromManifest(manifest)).rejects.toThrow('shape mismatch');
    });

    it('should skip unrecognized tensor names', async () => {
      const manifest = createTestLoRAManifest();
      manifest.tensors.push({
        name: 'unrecognized.tensor.name',
        shape: [2, 2],
        data: [1, 2, 3, 4],
      });

      const adapter = await loadLoRAFromManifest(manifest);
      // Should not throw, just skip the unrecognized tensor
      expect(adapter.layers.size).toBe(1);
    });

    it('should call progress callback', async () => {
      const manifest = createTestLoRAManifest();
      const progress: number[] = [];

      await loadLoRAFromManifest(manifest, {
        onProgress: (loaded, total) => {
          progress.push(loaded / total);
        },
      });

      expect(progress.length).toBeGreaterThan(0);
      expect(progress[progress.length - 1]).toBe(1);
    });
  });
});

// ============================================================================
// Adapter Manager Tests
// ============================================================================

describe('AdapterManager', () => {
  let manager: AdapterManager;

  beforeEach(() => {
    manager = new AdapterManager();
  });

  afterEach(() => {
    manager.unloadAll();
    resetAdapterManager();
  });

  describe('registerAdapter', () => {
    it('should register an adapter', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);

      const state = manager.registerAdapter('test', adapter, createTestManifest());

      expect(state.id).toBe('test');
      expect(state.enabled).toBe(false);
      expect(manager.isLoaded('test')).toBe(true);
    });

    it('should throw if adapter already registered', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);

      manager.registerAdapter('test', adapter, createTestManifest());

      expect(() =>
        manager.registerAdapter('test', adapter, createTestManifest())
      ).toThrow('already loaded');
    });
  });

  describe('enableAdapter/disableAdapter', () => {
    it('should enable and disable adapter', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);
      manager.registerAdapter('test', adapter, createTestManifest());

      expect(manager.isEnabled('test')).toBe(false);

      manager.enableAdapter('test');
      expect(manager.isEnabled('test')).toBe(true);
      expect(manager.getActiveAdapterIds()).toContain('test');

      manager.disableAdapter('test');
      expect(manager.isEnabled('test')).toBe(false);
      expect(manager.getActiveAdapterIds()).not.toContain('test');
    });

    it('should throw if adapter not found', () => {
      expect(() => manager.enableAdapter('nonexistent')).toThrow('not found');
      expect(() => manager.disableAdapter('nonexistent')).toThrow('not found');
    });

    it('should validate base model if requested', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);
      manager.registerAdapter('test', adapter, createTestManifest({ baseModel: 'model-a' }));

      expect(() =>
        manager.enableAdapter('test', {
          validateBaseModel: true,
          expectedBaseModel: 'model-b',
        })
      ).toThrow('base model');
    });

    it('should support weight multiplier', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);
      manager.registerAdapter('test', adapter, createTestManifest());

      manager.enableAdapter('test', { weight: 0.5 });

      const state = manager.getAdapterState('test');
      expect(state?.weight).toBe(0.5);
    });

    it('should reject invalid weight', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);
      manager.registerAdapter('test', adapter, createTestManifest());

      expect(() => manager.enableAdapter('test', { weight: 3.0 })).toThrow('between 0.0 and 2.0');
    });
  });

  describe('toggleAdapter', () => {
    it('should toggle adapter state', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);
      manager.registerAdapter('test', adapter, createTestManifest());

      expect(manager.toggleAdapter('test')).toBe(true);
      expect(manager.isEnabled('test')).toBe(true);

      expect(manager.toggleAdapter('test')).toBe(false);
      expect(manager.isEnabled('test')).toBe(false);
    });
  });

  describe('getActiveAdapter', () => {
    it('should return null when no adapters active', () => {
      expect(manager.getActiveAdapter()).toBeNull();
    });

    it('should return single active adapter', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);
      manager.registerAdapter('test', adapter, createTestManifest());
      manager.enableAdapter('test');

      const active = manager.getActiveAdapter();
      expect(active).not.toBeNull();
      expect(active?.name).toBe('test-lora');
    });

    it('should merge multiple active adapters', async () => {
      const manifest1 = createTestLoRAManifest();
      const adapter1 = await loadLoRAFromManifest(manifest1);
      manager.registerAdapter('test1', adapter1, createTestManifest({ id: 'test1' }));

      const manifest2 = createTestLoRAManifest();
      manifest2.name = 'test-lora-2';
      const adapter2 = await loadLoRAFromManifest(manifest2);
      manager.registerAdapter('test2', adapter2, createTestManifest({ id: 'test2' }));

      manager.enableAdapter('test1');
      manager.enableAdapter('test2');

      const active = manager.getActiveAdapter();
      expect(active).not.toBeNull();
      expect(active?.name).toContain('merged');
    });
  });

  describe('disableAll', () => {
    it('should disable all adapters', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);

      manager.registerAdapter('test1', adapter, createTestManifest({ id: 'test1' }));
      manager.registerAdapter('test2', adapter, createTestManifest({ id: 'test2' }));

      manager.enableAdapter('test1');
      manager.enableAdapter('test2');

      expect(manager.enabledCount).toBe(2);

      manager.disableAll();

      expect(manager.enabledCount).toBe(0);
    });
  });

  describe('enableOnly', () => {
    it('should enable only specified adapter', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);

      manager.registerAdapter('test1', adapter, createTestManifest({ id: 'test1' }));
      manager.registerAdapter('test2', adapter, createTestManifest({ id: 'test2' }));

      manager.enableAdapter('test1');
      manager.enableAdapter('test2');

      manager.enableOnly('test1');

      expect(manager.isEnabled('test1')).toBe(true);
      expect(manager.isEnabled('test2')).toBe(false);
    });
  });

  describe('unloadAdapter', () => {
    it('should unload adapter', async () => {
      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);
      manager.registerAdapter('test', adapter, createTestManifest());
      manager.enableAdapter('test');

      manager.unloadAdapter('test');

      expect(manager.isLoaded('test')).toBe(false);
      expect(manager.isEnabled('test')).toBe(false);
    });
  });

  describe('events', () => {
    it('should fire events on state changes', async () => {
      const events: string[] = [];

      manager.setEvents({
        onAdapterLoaded: (id) => events.push(`loaded:${id}`),
        onAdapterEnabled: (id) => events.push(`enabled:${id}`),
        onAdapterDisabled: (id) => events.push(`disabled:${id}`),
        onAdapterUnloaded: (id) => events.push(`unloaded:${id}`),
      });

      const manifest = createTestLoRAManifest();
      const adapter = await loadLoRAFromManifest(manifest);

      manager.registerAdapter('test', adapter, createTestManifest());
      manager.enableAdapter('test');
      manager.disableAdapter('test');
      manager.unloadAdapter('test');

      expect(events).toEqual([
        'loaded:test',
        'enabled:test',
        'disabled:test',
        'unloaded:test',
      ]);
    });
  });
});

// ============================================================================
// Adapter Registry Tests
// ============================================================================

describe('AdapterRegistry', () => {
  let registry: AdapterRegistry;

  beforeEach(() => {
    registry = createMemoryRegistry();
  });

  afterEach(async () => {
    await registry.clear();
    resetAdapterRegistry();
  });

  describe('register', () => {
    it('should register adapter', async () => {
      const manifest = createTestManifest();

      const entry = await registry.register(manifest, {
        storageType: 'url',
        manifestPath: 'https://example.com/adapter.json',
      });

      expect(entry.id).toBe(manifest.id);
      expect(entry.name).toBe(manifest.name);
      expect(entry.storageType).toBe('url');
    });

    it('should reject invalid manifest', async () => {
      await expect(
        registry.register({} as any, { storageType: 'url', manifestPath: 'test' })
      ).rejects.toThrow('Invalid manifest');
    });
  });

  describe('unregister', () => {
    it('should remove entry', async () => {
      const manifest = createTestManifest();
      await registry.register(manifest, { storageType: 'url', manifestPath: 'test' });

      expect(await registry.has(manifest.id)).toBe(true);

      const result = await registry.unregister(manifest.id);
      expect(result).toBe(true);

      expect(await registry.has(manifest.id)).toBe(false);
    });

    it('should return false if not found', async () => {
      const result = await registry.unregister('nonexistent');
      expect(result).toBe(false);
    });
  });

  describe('get', () => {
    it('should return entry by id', async () => {
      const manifest = createTestManifest();
      await registry.register(manifest, { storageType: 'url', manifestPath: 'test' });

      const entry = await registry.get(manifest.id);
      expect(entry).not.toBeNull();
      expect(entry?.name).toBe(manifest.name);
    });

    it('should return null if not found', async () => {
      const entry = await registry.get('nonexistent');
      expect(entry).toBeNull();
    });

    it('should update lastAccessedAt', async () => {
      const manifest = createTestManifest();
      await registry.register(manifest, { storageType: 'url', manifestPath: 'test' });

      const before = (await registry.get(manifest.id))?.lastAccessedAt;

      // Wait a bit
      await new Promise(r => setTimeout(r, 10));

      const after = (await registry.get(manifest.id))?.lastAccessedAt;

      expect(after).toBeGreaterThan(before!);
    });
  });

  describe('list', () => {
    beforeEach(async () => {
      await registry.register(createTestManifest({ id: 'a', name: 'Alpha', baseModel: 'model-1' }), {
        storageType: 'url',
        manifestPath: 'a.json',
      });
      await registry.register(createTestManifest({ id: 'b', name: 'Beta', baseModel: 'model-1' }), {
        storageType: 'url',
        manifestPath: 'b.json',
      });
      await registry.register(createTestManifest({ id: 'c', name: 'Gamma', baseModel: 'model-2' }), {
        storageType: 'url',
        manifestPath: 'c.json',
      });
    });

    it('should list all entries', async () => {
      const entries = await registry.list();
      expect(entries).toHaveLength(3);
    });

    it('should filter by baseModel', async () => {
      const entries = await registry.list({ baseModel: 'model-1' });
      expect(entries).toHaveLength(2);
      expect(entries.every(e => e.baseModel === 'model-1')).toBe(true);
    });

    it('should sort by name', async () => {
      const asc = await registry.list({ sortBy: 'name', sortOrder: 'asc' });
      expect(asc[0].name).toBe('Alpha');
      expect(asc[2].name).toBe('Gamma');

      const desc = await registry.list({ sortBy: 'name', sortOrder: 'desc' });
      expect(desc[0].name).toBe('Gamma');
      expect(desc[2].name).toBe('Alpha');
    });

    it('should apply limit and offset', async () => {
      const page1 = await registry.list({ limit: 2, offset: 0 });
      expect(page1).toHaveLength(2);

      const page2 = await registry.list({ limit: 2, offset: 2 });
      expect(page2).toHaveLength(1);
    });
  });

  describe('count', () => {
    it('should count entries', async () => {
      await registry.register(createTestManifest({ id: 'a' }), { storageType: 'url', manifestPath: 'a' });
      await registry.register(createTestManifest({ id: 'b' }), { storageType: 'url', manifestPath: 'b' });

      expect(await registry.count()).toBe(2);
    });

    it('should count with filter', async () => {
      await registry.register(createTestManifest({ id: 'a', baseModel: 'm1' }), { storageType: 'url', manifestPath: 'a' });
      await registry.register(createTestManifest({ id: 'b', baseModel: 'm2' }), { storageType: 'url', manifestPath: 'b' });

      expect(await registry.count({ baseModel: 'm1' })).toBe(1);
    });
  });

  describe('getBaseModels', () => {
    it('should return unique base models', async () => {
      await registry.register(createTestManifest({ id: 'a', baseModel: 'model-1' }), { storageType: 'url', manifestPath: 'a' });
      await registry.register(createTestManifest({ id: 'b', baseModel: 'model-1' }), { storageType: 'url', manifestPath: 'b' });
      await registry.register(createTestManifest({ id: 'c', baseModel: 'model-2' }), { storageType: 'url', manifestPath: 'c' });

      const models = await registry.getBaseModels();
      expect(models).toEqual(['model-1', 'model-2']);
    });
  });

  describe('updateMetadata', () => {
    it('should update metadata', async () => {
      await registry.register(createTestManifest({ id: 'test' }), { storageType: 'url', manifestPath: 'test' });

      const updated = await registry.updateMetadata('test', { author: 'Test Author' });

      expect(updated?.metadata?.author).toBe('Test Author');
    });

    it('should return null if not found', async () => {
      const result = await registry.updateMetadata('nonexistent', {});
      expect(result).toBeNull();
    });
  });

  describe('exportToJSON/importFromJSON', () => {
    it('should export and import entries', async () => {
      await registry.register(createTestManifest({ id: 'a' }), { storageType: 'url', manifestPath: 'a' });
      await registry.register(createTestManifest({ id: 'b' }), { storageType: 'url', manifestPath: 'b' });

      const json = await registry.exportToJSON();

      // Clear and reimport
      await registry.clear();
      expect(await registry.count()).toBe(0);

      const result = await registry.importFromJSON(json);

      expect(result.imported).toBe(2);
      expect(result.skipped).toBe(0);
      expect(await registry.count()).toBe(2);
    });

    it('should skip existing entries without overwrite', async () => {
      await registry.register(createTestManifest({ id: 'a' }), { storageType: 'url', manifestPath: 'a' });

      const json = JSON.stringify([{ ...createTestManifest({ id: 'a' }), storageType: 'url', manifestPath: 'a', registeredAt: 0, lastAccessedAt: 0 }]);

      const result = await registry.importFromJSON(json);

      expect(result.imported).toBe(0);
      expect(result.skipped).toBe(1);
    });

    it('should overwrite with flag', async () => {
      await registry.register(createTestManifest({ id: 'a', name: 'Original' }), { storageType: 'url', manifestPath: 'a' });

      const entry = {
        ...createTestManifest({ id: 'a', name: 'Updated' }),
        storageType: 'url' as const,
        manifestPath: 'a',
        registeredAt: Date.now(),
        lastAccessedAt: Date.now(),
      };
      const json = JSON.stringify([entry]);

      await registry.importFromJSON(json, { overwrite: true });

      const imported = await registry.get('a');
      expect(imported?.name).toBe('Updated');
    });
  });
});
