import { describe, it, expect, afterEach, vi } from 'vitest';
import { saveReport } from '../../src/storage/reports.js';

let lastWrite = null;

vi.mock('../../src/storage/backends/idb-store.js', () => ({
  createIdbStore: () => ({
    openModel: async () => {},
    writeFile: async (filename, payload) => {
      lastWrite = { filename, payload };
    },
    cleanup: async () => {},
  }),
}));

vi.mock('../../src/storage/quota.js', () => ({
  isOPFSAvailable: () => false,
  isIndexedDBAvailable: () => true,
}));

describe('storage reports', () => {
  afterEach(() => {
    lastWrite = null;
    vi.clearAllMocks();
  });

  it('saves report payloads to indexeddb backend', async () => {
    const report = { suite: 'inference', results: [] };
    const result = await saveReport('test-model', report, { timestamp: '2026-01-01T00:00:00Z' });

    expect(result.backend).toBe('indexeddb');
    expect(result.path).toContain('reports/test-model/');
    expect(lastWrite).not.toBeNull();
    expect(lastWrite.filename).toBe('2026-01-01T00-00-00Z.json');
    expect(JSON.parse(lastWrite.payload).suite).toBe('inference');
  });
});
