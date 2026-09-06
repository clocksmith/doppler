import type { ElectronReleaseStateCoordinator, ElectronReleaseStateStore, ElectronRevocationSnapshot } from 'doppler-gpu/electron';

export declare function registerDocumentSearchReleaseMain(options: {
  stateStore: ElectronReleaseStateStore;
  verifyReleaseDecision(decision: Record<string, unknown>): Promise<boolean> | boolean;
  verifyRevocationSnapshot(snapshot: ElectronRevocationSnapshot): Promise<boolean> | boolean;
  authorizeRequest(event: unknown, request: Record<string, unknown> & { action: string }): Promise<boolean> | boolean;
  now?: () => string;
  ipcMain: { handle(channel: string, handler: (...args: unknown[]) => unknown): void };
}): ElectronReleaseStateCoordinator;
