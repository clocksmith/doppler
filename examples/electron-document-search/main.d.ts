import type { ElectronReleaseStateCoordinator } from '../../src/client/electron/index.js';

export declare function registerDocumentSearchReleaseMain(options: {
  stateStore: Record<string, unknown>;
  verifyReleaseDecision(decision: Record<string, unknown>): Promise<boolean> | boolean;
  verifyRevocationSnapshot(snapshot: Record<string, unknown>): Promise<boolean> | boolean;
  now?: () => string;
  ipcMain: { handle(channel: string, handler: (...args: unknown[]) => unknown): void };
}): ElectronReleaseStateCoordinator;
